import json

import numpy as np
import pandas as pd
import pytest
import torch
from lifelines.utils import concordance_index

from src.data import CLINICAL_FEATURES, DERIVED, FORBIDDEN_PREDICTORS, ROOT, clinical_features, cv_folds, patient_from_report, split_for, stage_ordinal
from src.features import BlockPCA, ClinicalPreprocessor, fit_transform_blocks
from src.models import BreslowLoss, LateFusionCox, concordance, inner_split, paired_bootstrap

HAS_DATA = (DERIVED / "clinical.csv").exists() and (ROOT / "data" / "membership.json").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run `python -m src.data` first")
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())


# ------------------------------------------------------------------ patient-split integrity
def test_split_is_deterministic_stratified_and_disjoint():
    ids = [f"TCGA-XX-{i:04d}" for i in range(400)]
    ev = [int(i % 7 == 0) for i in range(400)]
    a, b = split_for(ids, ev), split_for(list(reversed(ids)), list(reversed(ev)))
    assert a == b  # independent of row order
    df = pd.DataFrame({"id": ids, "ev": ev, "s": [a[i] for i in ids]})
    for _, g in df.groupby("ev"):
        assert abs((g.s == "test").mean() - 0.25) < 0.01
    with pytest.raises(ValueError):
        split_for(ids + ids[:1], ev + ev[:1])


def test_cv_folds_cover_each_patient_once_and_balance_events():
    ids = [f"TCGA-XX-{i:04d}" for i in range(250)]
    ev = [int(i % 5 == 0) for i in range(250)]
    f = cv_folds(ids, ev, repeat=0)
    assert set(f) == set(ids) and set(f.values()) == {0, 1, 2, 3, 4}
    counts = pd.Series({i: f[i] for i, e in zip(ids, ev) if e}).value_counts()
    assert counts.max() - counts.min() <= 1
    assert f != cv_folds(ids, ev, repeat=1)


@needs_data
def test_saved_membership_matches_rule_and_label_files():
    membership = json.loads((ROOT / "data" / "membership.json").read_text())
    dev, test = pd.read_csv(DERIVED / "dev_labels.csv"), pd.read_csv(DERIVED / "test_labels.csv")
    assert not set(dev.patient_id) & set(test.patient_id)
    assert dev.patient_id.is_unique and test.patient_id.is_unique
    both = pd.concat([dev, test])
    assert split_for(both.patient_id, both.pfi_event) == membership
    assert (dev.patient_id.map(membership) == "dev").all() and (test.patient_id.map(membership) == "test").all()


@needs_data
def test_dev_loader_never_returns_test_labels():
    from src.run import load_inputs

    if not (ROOT / "data" / "embeddings" / "embedding_meta.json").exists():
        pytest.skip("embeddings not computed")
    _, clin, _, _, labels, rows = load_inputs("dev")
    assert (clin.split.iloc[rows] == "dev").all()
    test_ids = set(pd.read_csv(DERIVED / "test_labels.csv").patient_id)
    assert not test_ids & set(labels.patient_id)


def test_inner_validation_split_is_patient_disjoint():
    ids = [f"p{i}" for i in range(100)]
    fit, val = inner_split(ids, np.arange(100) % 4 == 0, 0.2)
    assert not (fit & val).any() and (fit | val).all() and 18 <= val.sum() <= 22


# ------------------------------------------------------------------ training-only preprocessing
def test_clinical_preprocessor_uses_training_statistics_only():
    tr = np.array([[50.0, 1.0], [60.0, np.nan], [70.0, 3.0]])
    p = ClinicalPreprocessor([1]).fit(tr)
    assert p.median.tolist() == [60.0, 2.0]
    out = p.transform(np.array([[1000.0, np.nan]]))
    assert out.shape == (1, 3)
    # imputed with TRAINING median (2.0) -> standardized with training stats
    assert out[0, 1] == pytest.approx((2.0 - p.mean[1]) / p.scale[1])
    assert out[0, 2] > 0  # missing indicator set


def test_block_pca_selects_genes_and_components_from_training_rows_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 50)).astype(np.float32)
    x[:, 3] *= 10  # high variance everywhere -> should be kept
    x[40:, 7] = rng.normal(scale=1000, size=20)  # extreme variance only in evaluation rows
    tr, ev = np.arange(40), np.arange(40, 60)
    p = BlockPCA(4, top_variance=10).fit(x[tr])
    assert 3 in p.columns and 7 not in p.columns


def test_fit_transform_blocks_train_output_independent_of_eval_rows():
    rng = np.random.default_rng(1)
    raw = {"clinical": rng.normal(size=(80, 4)), "rna": rng.normal(size=(80, 30)).astype(np.float32), "text": rng.normal(size=(80, 20)).astype(np.float32)}
    raw["clinical"][::9, 1] = np.nan
    proto = {**PROTOCOL, "modalities": {"rna": {"top_variance_genes": 10, "pca_components": 3}, "text": {"pca_components": 3}}}
    tr, ev = np.arange(60), np.arange(60, 80)
    a, _ = fit_transform_blocks(raw, tr, ev, proto)
    poisoned = {k: v.copy() for k, v in raw.items()}
    for v in poisoned.values():
        v[ev] = 1e4
    b, _ = fit_transform_blocks(poisoned, tr, ev, proto)
    for m in raw:
        np.testing.assert_allclose(a[m][0], b[m][0])
    with pytest.raises(ValueError):
        fit_transform_blocks(raw, tr, np.arange(50, 70), proto)


# ------------------------------------------------------------------ join correctness
def test_report_filename_parsing_handles_both_observed_formats():
    assert patient_from_report("TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0") == "TCGA-BP-5195"
    assert patient_from_report("TCGA-B0-5116.ff5d6b2fb5282d2487471f65f0ae8d0e") == "TCGA-B0-5116"
    with pytest.raises(ValueError):
        patient_from_report("TCGA-BP-519.abc")


def test_stage_and_histology_encoding():
    assert [stage_ordinal(s) for s in ["Stage I", "Stage IA", "Stage IIB", "Stage IIIC", "Stage IV"]] == [1, 1, 2, 3, 4]
    assert all(np.isnan(stage_ordinal(s)) for s in ["Stage X", "[Discrepancy]", "[Not Available]", None])
    cdr = pd.DataFrame({"bcr_patient_barcode": ["a", "b", "c"], "age_at_initial_pathologic_diagnosis": [50, 60, None], "ajcc_pathologic_tumor_stage": ["Stage IIA", "Stage X", "Stage IV"], "histological_type": ["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Mucinous Carcinoma"]})
    out = clinical_features(cdr)
    assert out.histology_lobular.tolist() == [0, 1, 0] and out.histology_other.tolist() == [0, 0, 1]


def test_no_forbidden_predictor_reaches_feature_matrix():
    assert PROTOCOL["clinical_features"] == CLINICAL_FEATURES
    assert not set(PROTOCOL["clinical_features"]) & set(FORBIDDEN_PREDICTORS)
    assert set(PROTOCOL["missing_indicator_features"]) <= set(CLINICAL_FEATURES)


@needs_data
def test_derived_tables_are_aligned_by_patient_and_match_cdr():
    from src.data import load_cdr_brca

    clin = pd.read_csv(DERIVED / "clinical.csv")
    rep = pd.read_csv(DERIVED / "reports.csv", usecols=["patient_id", "patient_filename"])
    rna = json.loads((DERIVED / "rna_genes.json").read_text())
    assert clin.patient_id.is_unique
    assert clin.patient_id.tolist() == rep.patient_id.tolist() == [s[:12] for s in rna["samples"]]
    assert all(s[13:15] == "01" for s in rna["samples"])
    assert (rep.patient_filename.str.split(".").str[0] == rep.patient_id).all()
    assert np.load(DERIVED / "rna.npy", mmap_mode="r").shape == (len(clin), len(rna["genes"]))
    labels = pd.concat([pd.read_csv(DERIVED / f"{p}_labels.csv") for p in ["dev", "test"]]).set_index("patient_id")
    cdr = load_cdr_brca().set_index("bcr_patient_barcode").loc[clin.patient_id]
    np.testing.assert_array_equal(labels.loc[clin.patient_id, "pfi_time"].to_numpy(), cdr["PFI.time"].to_numpy())
    np.testing.assert_array_equal(labels.loc[clin.patient_id, "pfi_event"].to_numpy(), cdr["PFI"].to_numpy())
    np.testing.assert_array_equal(clin.age.to_numpy(), cdr.age_at_initial_pathologic_diagnosis.to_numpy())
    assert cdr.Redaction.isna().all() and (cdr["PFI.time"] > 0).all()


@needs_data
def test_rna_values_match_raw_matrix_for_a_spot_check():
    import gzip

    from src.data import RAW

    rna = json.loads((DERIVED / "rna_genes.json").read_text())
    mat = np.load(DERIVED / "rna.npy", mmap_mode="r")
    gene = rna["genes"][0]
    with gzip.open(RAW / "HiSeqV2.gz", "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == gene:
                break
    raw = dict(zip(header[1:], parts[1:]))
    for j in [0, len(rna["samples"]) // 2, len(rna["samples"]) - 1]:
        assert mat[j, 0] == pytest.approx(float(raw[rna["samples"][j]]), abs=1e-4)


# ------------------------------------------------------------------ text chunking
def test_chunking_overlaps_and_respects_512_limit():
    from src.text import MAX_LENGTH, MODEL_ID, MODEL_REVISION, STRIDE, chunk_report

    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")
    text = " ".join(f"carcinoma{i % 50} margin" for i in range(900))
    chunks = chunk_report(tok, text)
    assert len(chunks) > 1 and all(len(c) <= MAX_LENGTH for c in chunks)
    assert all(c[0] == tok.cls_token_id and c[-1] == tok.sep_token_id for c in chunks)
    first, second = chunks[0][1:-1], chunks[1][1:-1]
    assert first[-STRIDE:] == second[:STRIDE]
    body = tok(text, add_special_tokens=False)["input_ids"]
    covered = chunks[0][1:-1] + [t for c in chunks[1:] for t in c[1 + STRIDE:-1]]
    assert covered == body


def test_leakage_flags_detect_prespecified_language():
    from src.text import leakage_flags

    f = leakage_flags(["Patient expired; autopsy performed.", "Oncotype DX recurrence score 18.", "Invasive ductal carcinoma, grade 2."])
    assert f.death_or_autopsy.tolist() == [True, False, False]
    assert f.recurrence_score_assay.tolist() == [False, True, False]
    assert not f.iloc[2].any()


# ------------------------------------------------------------------ survival math
@pytest.mark.parametrize("risk", [[1.0, 2.0, -1.0, 0.5], [1000.0, 1001.0, 998.0, 999.5]])
def test_breslow_loss_equals_explicit_risk_sets(risk):
    t = torch.tensor([3.0, 3.0, 5.0, 7.0])
    e = torch.tensor([1.0, 1.0, 0.0, 1.0])
    r = torch.tensor(risk, requires_grad=True)
    brute = torch.stack([-(r[i] - torch.logsumexp(r[t >= t[i]], 0)) for i in range(4) if e[i]]).mean()
    actual = BreslowLoss(t, e)(r)
    assert torch.allclose(actual, brute, atol=1e-5)
    actual.backward()
    assert torch.isfinite(r.grad).all()


def test_concordance_matches_lifelines_without_ties_and_handles_censoring():
    rng = np.random.default_rng(7)
    t, e, r = rng.uniform(1, 30, 120), rng.integers(0, 2, 120), rng.normal(size=120)
    assert concordance(t, e, r) == pytest.approx(concordance_index(t, -r, e))
    assert concordance([1, 2, 3], [0, 1, 0], [100, 2, 1]) == 1  # early censored record forms no pair


def test_paired_bootstrap_identical_models_have_zero_difference():
    rng = np.random.default_rng(3)
    t, e, r = rng.uniform(1, 30, 80), rng.integers(0, 2, 80), rng.normal(size=80)
    rows, diffs, _ = paired_bootstrap(t, e, {"a": r, "b": r.copy(), "c": np.stack([r, -r])}, [("b", "a")], 200, 0)
    assert diffs["b - a"]["estimate"] == 0 and diffs["b - a"]["ci95"] == [0, 0]
    assert rows["c"]["c_index"] == pytest.approx(0.5)  # seed-averaged pair scores
    assert rows["a"]["ci95"][0] <= rows["a"]["c_index"] <= rows["a"]["ci95"][1]


def test_late_fusion_network_backpropagates_through_every_branch():
    m = LateFusionCox([5, 16, 16])
    risk = m([torch.randn(12, 5), torch.randn(12, 16), torch.randn(12, 16)])
    BreslowLoss(torch.arange(1.0, 13.0), torch.tensor([1.0, 0.0] * 6))(risk).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
