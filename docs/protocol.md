# Prespecified protocol (v1.0)

Machine-readable twin: [`../protocol.json`](../protocol.json). This file and `protocol.json` were committed **before any model was fit or evaluated**. The cohort counts below come from `data/manifest.json`, which was built before modelling. Any later change is listed under "Deviations" with its reason and timing.

## 1. Question

In TCGA-BRCA patients who have all three modalities, does a frozen biomedical-transformer embedding of the diagnostic pathology report improve held-out ranking of progression-free interval (PFI)? We test this against baseline clinical covariates, and separately against clinical covariates plus RNA-seq.

This is a retrospective research prototype built on public, de-identified data. It is not a clinical tool.

## 2. Data

| Modality | Source | Unit joined | License / terms |
|---|---|---|---|
| Pathology report text | TCGA-Reports (Kefeli & Tatonetti, *Patterns* 2024), `TCGA_Reports.csv.zip` from GitHub `jkefeli/tcga-path-reports` | patient barcode (first 12 chars of `patient_filename`) | Dataset CC BY 4.0 (Mendeley Data, DOI 10.17632/hyg5xkznpx.1). Repository code MIT. |
| Gene expression | UCSC Xena TCGA hub `TCGA.BRCA.sampleMap/HiSeqV2` (log2(RSEM norm_count+1), Illumina HiSeq RNASeqV2, Xena version 2017-10-13) | primary-tumor sample (`-01`), one per patient | Xena metadata: `security: public`, `redistribution: true`. Underlying data is TCGA open access. |
| Clinical covariates + endpoints | TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR; Liu et al., *Cell* 2018), `TCGA-CDR-SupplementalTableS1.xlsx`, sheet `TCGA-CDR`, from GDC | patient barcode | GDC open-access PanCanAtlas publication supplement |

URLs, retrieval dates and SHA-256 hashes are recorded in `data/manifest.json`. Raw files are gitignored. `python -m src.data` re-downloads them and checks their hashes.

### Cohort construction (counts from `data/manifest.json`)

1. TCGA-CDR rows with `type == "BRCA"`: **1097**
2. Excluded `Redaction == "Redacted"`: **1092**
3. Required `PFI.time > 0` and `OS.time > 0`, with known event status (13 rows had time 0, 1 had missing time): **1078**
4. Has a TCGA-Reports pathology report: **1019**
5. Also has primary-tumor RNA-seq: **1018** (the joint cohort used in every comparison)

The tumor-type label is taken from CDR `type`. The report repository's own cancer-type map is used only as a cross-check. There is at most one report per patient and at most one primary-tumor RNA sample per patient. The code raises an error if either assumption fails.

### Endpoint choice

- **Primary: PFI. Secondary: OS.** Both come from TCGA-CDR, with time in days.
- **Justification.** The TCGA-CDR notes sheet that ships with the table says: *"For clinical outcome endpoints, we recommend the use of PFI for progression-free interval, and OS for overall survival. Both endpoints are relatively accurate. Given the relatively short follow-up time, PFI is preferred over OS."* Table 3 of Liu et al. (2018) gives the detailed per-cancer recommendations.
- **Note on event counts.** In this joint cohort PFI has 140 events and OS has 145 (`data/manifest.json`), so OS does **not** have fewer events. The reason for preferring PFI is short follow-up relative to breast cancer's long natural history. With short follow-up, OS events are sparse in time and mix cancer and non-cancer deaths, while PFI captures earlier disease-related events.

## 3. Predictors

- **Clinical (4 features + 1 missing indicator):**
  - `age`: age at initial pathologic diagnosis.
  - `stage`: AJCC pathologic stage collapsed to I–IV and treated as ordinal. Stage X, `[Discrepancy]` and `[Not Available]` count as missing (24/1018). Missing values get the training median plus a missing indicator.
  - `histology_lobular` and `histology_other`: one-hot, with infiltrating ductal as the reference.
- **Excluded on purpose:**
  - Sex: 12 male patients, too few to estimate an effect.
  - Race: social construct with about 9% missing. We did not want to build it into a risk score.
  - Menopause status: largely redundant with age.
  - Clinical stage.
  - Every follow-up-derived or endpoint-derived field. The list is `FORBIDDEN_PREDICTORS` in `src/data.py`, and a test enforces it. It includes `initial_pathologic_dx_year`, because diagnosis year tracks follow-up length mechanically.
- **RNA:** all genes with no missing values in the joint cohort. Inside each training set only:
  1. Keep the 5000 highest-variance genes.
  2. Standardize.
  3. PCA to 16 components.
  4. Re-standardize the scores.
- **Text:** a 768-dimensional report embedding (section 4). Inside each training set only: standardize, PCA to 16 components, re-standardize.

## 4. Text encoder

- **Model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, pinned to revision `e1354b7a3a09615f6aba48dfad4b7a613eef7062`. Model-card license: MIT.
- **Frozen:** the encoder is never fine-tuned and gets no gradient updates. It runs on CPU under `torch.inference_mode()`.
- **Chunking:** the fast tokenizer splits each report into windows of at most 512 tokens (`return_overflowing_tokens=True, stride=64`). Consecutive windows share 64 tokens, and each window carries its own `[CLS]` and `[SEP]`.
- **Pooling:** each chunk's last hidden layer is mean-pooled using the attention mask. The report embedding is the unweighted mean of its chunk vectors.
- **Cache:** embeddings are written to `data/embeddings/report_embeddings.npy`, which is gitignored. Its hash is recorded.
- **OCR caveat:** TCGA-Reports text was produced by OCR (AWS Textract) from scanned PDFs, then cleaned. Character errors remain, for example "Recelpt" for "Receipt" and "turnor" for "tumor". Some reports are translated or non-US in style.

## 5. Leakage audit (text only, no outcomes)

Pathology reports are written around diagnosis and surgery, but they can still mention history or prognosis. `src/text.py` defines fixed regular-expression flags and runs them over all 1018 reports. Results go to `results/leakage_scan.json`. The pass reads text only, never labels.

What the pre-modelling scan found:

- No death or autopsy language and no relapse or progression language.
- 26 reports quote an Oncotype DX **recurrence score**. This is a diagnosis-time prognostic assay result, so it is legitimate baseline information, but it is strongly prognostic.
- 15 describe recurrent or prior breast cancer.
- 89 mention neoadjuvant or post-treatment changes.
- "Follow-up" appears only as "follow-up resection material", meaning re-excision at surgery.
- "Survival" appears in boilerplate assay text.

Policy:

- The primary analysis keeps every patient.
- A prespecified **sensitivity analysis** evaluates the already-fitted models on the test set after removing patients flagged by any of: death/autopsy, relapse/progression, recurrence-score assay, recurrent/prior breast cancer, or neoadjuvant/post-treatment. Models are not refit.
- **Dev-only site diagnostics, which do not use the test set:**
  - Can text PCs predict the tissue source site (TSS)? This step uses no outcomes.
  - How well does a site-only Cox model rank PFI? Report formatting differs by hospital, and follow-up also differs by hospital, so text could look "useful" partly because it identifies the site.

## 6. Splits

- **Test set:** within each PFI-event stratum, patients are sorted by `SHA256(patient_id | "tcga-text-v1")` and the first 25% go to test. The result is **dev 763 patients (105 PFI events), test 255 patients (35 PFI events)**. Event status is read only for this stratification. `data/membership.json` is committed, and a test recomputes it.
- **Development CV:** 3 repeats of 5-fold CV inside dev, patient-level, stratified by PFI event with a different hash salt per repeat. All preprocessing (imputation, scaling, gene selection, PCA) is refit inside every training fold.
- **Network early stopping:** a deterministic event-stratified 20% inner-validation subset of each training set. It never comes from the evaluation fold or the test set.

## 7. Models (fixed, no hyperparameter search)

- **Cox PH (primary family):** lifelines `CoxPHFitter(penalizer=0.1, l1_ratio=0)`, a ridge penalty on standardized inputs, with Efron ties.
- **Late-fusion network (secondary family):**
  - One branch per included modality: `Linear(d,16) → ReLU → Dropout(0.2) → Linear(16,8) → ReLU`. The branches are concatenated into `Linear(·,1, bias=False)`.
  - Loss: Breslow negative Cox partial log-likelihood, full batch.
  - Optimizer: Adam with lr 1e-3 and weight decay 1e-3.
  - Training: at most 400 epochs, patience 30, min_delta 1e-4. The weights from the best inner-validation-loss epoch are restored.
  - Seeds: 11, 23, 37, 41, 53. C-index is averaged over seeds. There is no risk ensembling and no best-seed selection.
- **Feature-set ablations** (both families): `clinical` (reference), `clinical_rna`, `clinical_text`, `text`, `clinical_rna_text`, plus `rna` for symmetry.

## 8. Evaluation

- **Primary metric:** strict-time Harrell C-index. Only pairs with strictly ordered observed times and an event at the earlier time are compared, and tied risks get half credit. A test checks it against lifelines when there are no ties.
- **Secondary metric:** Uno's IPCW C (scikit-survival), with the censoring distribution taken from the development set and τ = 1826 days.
- **Uncertainty:** 2000 paired patient-bootstrap resamples of the test set (seed 2026). Every model uses the same resamples, and percentile 95% CIs are reported for each C-index and for each paired difference against `clinical`. The difference `clinical_rna_text − clinical_rna` is also reported.
- **Development CV:** reported as mean ± SD of fold C-index and of fold-paired differences. CV folds are not independent, so these SDs describe spread, not formal CIs.

### Decision rule (fixed now)

- Text is reported as **adding detectable held-out information** only if the paired-bootstrap 95% CI of **C(clinical_text) − C(clinical)**, for the primary Cox family on PFI on the test set, lies entirely above 0.
- **C(clinical_rna_text) − C(clinical_rna)** is the key secondary contrast and is judged by the same rule.
- Otherwise the result is reported as **no detectable gain in this cohort**. Point estimates alone are not treated as evidence. Test events are about 35, so wide intervals are expected.

## 9. Freeze and one-shot test

1. `python -m src.run dev` runs repeated CV on dev only. The loader opens `dev_labels.csv` only.
2. `python -m src.run freeze` refuses to run if any frozen file is uncommitted. Otherwise it records the git commit and the SHA-256 of the code, protocol, lockfile, derived data and embeddings in `results/freeze.json`.
3. `python -m src.run test` refuses to run unless every hash matches `freeze.json` and `freeze.json` is committed. It writes `results/test_results.json` in exclusive-create mode, so it runs once.

## 10. Compute

- CPU only. No GPU, no paid API, no cloud service, no LLM API calls.
- The embedding step runs on 6 threads with batches of 8 chunks and saves partial results every 50 reports.
- The model runner uses 1 torch thread.
- Peak RSS is recorded in the results.

## Deviations from the original brief

- **PFI rationale.** The brief said OS "has few events" in BRCA. In this cohort OS has slightly *more* events than PFI. PFI is kept as primary on the TCGA-CDR authors' actual stated reason, which is short follow-up (section 2).
- **Additional `rna`-only ablation.** It was added for symmetry and is descriptive only.
- **Computational-only change to `src/run.py`.** Fold-level process-pool parallelization and peak-RSS recording were added after the protocol commit (02fb4db) and before any development run (61f88a3). Methods are unchanged.
- **Network diagnostics.** Per-run diagnostics (best epoch, stop epoch) were saved only in aggregate for the development phase, and per run for the test phase.
