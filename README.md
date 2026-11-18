# TCGA Multimodal Text Lab

This is an independent, reproducible research prototype. It asks whether **pathology-report text**, encoded with a **frozen biomedical transformer**, adds held-out survival-ranking information to **clinical covariates** and **RNA-seq** in the same patients. The data are public and de-identified (TCGA breast cancer, BRCA). A null result is reported as a null result.

This is retrospective research software. It is not a diagnostic tool, not clinically validated and not affiliated with any hospital. It uses no PHI, no GPU, no paid API and no LLM calls.

## Question

The cohort is TCGA-BRCA patients who have all three modalities. Does a BiomedBERT embedding of the diagnostic pathology report improve the C-index for progression-free interval (PFI) beyond (a) age, stage and histology, and (b) clinical + RNA?

The full prespecified design is in [`docs/protocol.md`](docs/protocol.md). It was committed before any model was fit.

## Data

| Modality | Source | License / terms | N eligible (with clinical + endpoint) |
|---|---|---|---|
| Pathology report text | TCGA-Reports, Kefeli & Tatonetti, *Patterns* 2024 ([GitHub](https://github.com/jkefeli/tcga-path-reports), [Mendeley DOI 10.17632/hyg5xkznpx.1](https://doi.org/10.17632/hyg5xkznpx.1)) | CC BY 4.0 dataset; MIT code | 1019 |
| Gene expression | UCSC Xena `TCGA.BRCA.sampleMap/HiSeqV2`, log2(RSEM+1), primary tumor `-01` | Xena metadata `security: public`, `redistribution: true`; TCGA open access | 1076 |
| Clinical + endpoints | TCGA-CDR, Liu et al., *Cell* 2018 (GDC PanCanAtlas supplement) | GDC open-access publication data | 1078 (from 1097 BRCA rows; 5 redacted and 14 with zero or missing time removed) |
| **All three (analysis cohort)** | joined on the 12-character patient barcode | | **1018** (PFI events 140, OS events 145) |

Split (patient-level, stratified by PFI event, deterministic hash):

| Set | Patients | PFI events |
|---|---|---|
| Development | 763 | 105 |
| Test (used once) | 255 | 35 |

URLs, retrieval dates and SHA-256 hashes are in [`data/manifest.json`](data/manifest.json). Raw data are not committed.

**Endpoints.** The primary endpoint is PFI and the secondary is OS. The TCGA-CDR authors say "given the relatively short follow-up time, PFI is preferred over OS". In this cohort OS has slightly more events than PFI, so the reason for preferring PFI is short follow-up, not event count.

## Method

```mermaid
flowchart LR
  R[Pathology report, OCR text] --> T[Frozen BiomedBERT<br/>512-token chunks, stride 64<br/>mean-pool -> 768-d]
  G[RNA-seq 20k genes] --> P1[train-only: top-5000 variance<br/>scale, PCA-16]
  T --> P2[train-only: scale, PCA-16]
  C[Age, stage, histology] --> P0[train-only: impute, scale]
  P0 & P1 & P2 --> M[Penalized Cox PH  |  late-fusion Cox net]
  M --> E[Harrell C + paired patient bootstrap<br/>one-shot held-out test]
```

- **Text encoder:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` at a pinned revision, MIT license. The encoder is **frozen and not fine-tuned**. Long reports are tokenized into overlapping 512-token windows. Each window is mean-pooled with the attention mask, and the window vectors are averaged. The cached embeddings took 800 s on 6 CPU threads with about 1.5 GiB peak RSS.
- **Training-only preprocessing:** imputation, scaling, gene selection and PCA are refit inside every training fold and on the full development set. They never see evaluation rows, and a test checks this.
- **Models:**
  - Primary: ridge-penalized Cox PH (lifelines, penalizer 0.1).
  - Secondary: a small PyTorch late-fusion network with one branch per modality and a Breslow Cox partial-likelihood loss, trained with 5 seeds.
- **Ablations:** clinical, clinical+RNA, clinical+text, text, clinical+RNA+text, plus RNA.
- **Evaluation workflow:**
  1. 3×5-fold CV on the development set only.
  2. `freeze` records the git commit and hashes of code, data and embeddings.
  3. A single test run computes strict-time Harrell C, Uno's C (τ = 5 years) and 2000 paired patient-bootstrap CIs.
- **Decision rule, fixed in advance:** text "helps" only if the Cox PFI CI for C(clinical+text) − C(clinical) lies entirely above 0.

## Results (held-out test, PFI, n=255, 35 events)

| Feature set | Cox C [95% CI] | Δ vs clinical [95% CI] | Late-fusion NN C [95% CI] |
|---|---|---|---|
| clinical | 0.736 [0.633, 0.834] | reference | 0.634 [0.573, 0.691] |
| clinical + RNA | 0.733 [0.630, 0.818] | −0.003 [−0.083, 0.074] | 0.644 [0.563, 0.720] |
| clinical + text | 0.726 [0.632, 0.807] | −0.010 [−0.078, 0.065] | 0.707 [0.630, 0.775] |
| text only | 0.601 [0.499, 0.699] | −0.135 [−0.282, 0.013] | 0.622 [0.519, 0.715] |
| clinical + RNA + text | 0.733 [0.648, 0.807] | −0.003 [−0.084, 0.087] | 0.725 [0.646, 0.800] |
| RNA only | 0.613 [0.509, 0.705] | −0.123 [−0.256, 0.008] | 0.587 [0.489, 0.685] |

Key secondary contrast (Cox): clinical+RNA+text − clinical+RNA = +0.000 [−0.040, 0.042].

**Did text help? No detectable gain.** The primary contrast's CI spans 0, and so does the key secondary contrast. Removing 41 test patients whose reports carry prespecified outcome-adjacent language does not change this: Δ = −0.004 [−0.078, 0.082].

The network shows nominally positive text contrasts. These are not credible evidence:

- Its clinical-only reference is degenerate. In 2 of 5 seeds, early stopping kept the epoch-0 weights.
- Its best model (0.725) does not beat the clinical Cox baseline (0.736).

See [`reports/report.md`](reports/report.md) for these details, plus OS, development CV, Uno's C and the site-confound diagnostics. Text PCs identify the hospital with 89% accuracy, but site alone does not rank PFI (C 0.47).

## Reproduce

```bash
uv sync
uv run python -m src.data          # download (hash-checked) and patient-level join -> data/manifest.json
uv run python -m src.text scan     # text-only outcome-language audit -> results/leakage_scan.json
uv run python -m src.text embed    # frozen BiomedBERT embeddings (CPU, ~15 min)
uv run pytest -q                   # split integrity, train-only preprocessing, join correctness, survival math
uv run python -m src.run dev       # 3x5 CV on development split only
uv run python -m src.run freeze    # requires committed inputs; writes results/freeze.json
uv run python -m src.run test      # one shot; refuses if inputs changed or already run
uv run python scripts/render_tables.py results/tables.md
```

## Limitations

- **OCR text.** Reports were transcribed from scanned PDFs by OCR, and character errors remain. Some reports appear to be translated or non-US in style.
- **Site and format confound.** Text embeddings mostly encode institution and formatting (tissue-source-site accuracy 0.89).
- **Single cohort, retrospective, no external validation.** TCGA is not a consecutive clinical population, and follow-up is short for breast cancer.
- **Small number of events.** 35 test events give wide CIs, and C-index differences of a few points cannot be resolved.
- **Ranking only.** C-index says nothing about calibration or clinical benefit.
- **Frozen, general-purpose encoder.** No fine-tuning, no section-aware or structured extraction. The PCA-16 compression may discard useful signal.
- **Network weakness.** The fusion network's early stopping can keep untrained weights, as happened in the clinical-only reference. This was found after testing and was not fixed post hoc.
- **Not for clinical use.** This is a research prototype only.

## Repository map

- `src/data.py`: downloads, provenance, eligibility, patient join, split.
- `src/text.py`: chunking, frozen encoder, leakage audit.
- `src/features.py`: training-only preprocessing.
- `src/models.py`: Cox PH, late-fusion network, Breslow loss, C-index, paired bootstrap.
- `src/run.py`: dev / freeze / test phases.
- `results/`: all numbers.
- `reports/report.md`: write-up.
- `docs/protocol.md`: prespecified design.


