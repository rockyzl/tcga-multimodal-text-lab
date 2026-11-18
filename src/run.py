"""Phases: dev (repeated CV on development split only) -> freeze -> test (one shot)."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from src.data import DERIVED, ROOT, cv_folds, digest
from src.features import BlockPCA, fit_transform_blocks
from src.models import concordance, fit_cox, paired_bootstrap, predict_cox, predict_nn, train_nn, uno_c
from src.text import EMB

RESULTS = ROOT / "results"
ENDPOINTS = {"PFI": ("pfi_time", "pfi_event"), "OS": ("os_time", "os_event")}
FROZEN_FILES = ["protocol.json", "pyproject.toml", "uv.lock", "src/__init__.py", "src/data.py", "src/text.py", "src/features.py", "src/models.py", "src/run.py", "data/manifest.json", "data/membership.json"]


def protocol():
    return json.loads((ROOT / "protocol.json").read_text())


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x" if exclusive else "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def load_inputs(part):
    """Features for all joint-cohort patients (no outcomes) + labels for ONE requested part."""
    p = protocol()
    clin = pd.read_csv(DERIVED / "clinical.csv")
    reports = pd.read_csv(DERIVED / "reports.csv", usecols=["patient_id", "split", "tss"])
    order = clin.patient_id.tolist()
    if reports.patient_id.tolist() != order:
        raise ValueError("Clinical/report order mismatch")
    emb_meta = json.loads((EMB / "embedding_meta.json").read_text())
    if emb_meta["patient_order_sha256"] != __import__("hashlib").sha256("\n".join(order).encode()).hexdigest():
        raise ValueError("Embedding patient order mismatch")
    rna_meta = json.loads((DERIVED / "rna_genes.json").read_text())
    if [s[:12] for s in rna_meta["samples"]] != order:
        raise ValueError("RNA patient order mismatch")
    raw = {"clinical": clin[p["clinical_features"]].to_numpy(float), "rna": np.load(DERIVED / "rna.npy"), "text": np.load(EMB / "report_embeddings.npy")}
    labels = pd.read_csv(DERIVED / f"{part}_labels.csv")
    idx = pd.Series(range(len(order)), index=order)
    rows = idx.loc[labels.patient_id].to_numpy()
    if not (clin.split.iloc[rows] == part).all():
        raise ValueError("Label file contains patients from another split")
    return p, clin, reports, raw, labels, rows


def env_info():
    return {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count()}


def fit_predict_all(p, blocks_tr, blocks_ev, t_tr, e_tr, ids_tr, families=("cox", "nn")):
    """Return dict (family, feature_set) -> prediction array ((seeds,n) for nn) and training diagnostics."""
    preds, diag = {}, []
    for fs, mods in p["feature_sets"].items():
        xtr = [blocks_tr[m] for m in mods]
        xev = [blocks_ev[m] for m in mods]
        if "cox" in families:
            cph = fit_cox(np.concatenate(xtr, 1), t_tr, e_tr, p["cox"]["penalizer"], p["cox"]["l1_ratio"])
            preds[("cox", fs)] = predict_cox(cph, np.concatenate(xev, 1))
        if "nn" in families:
            seed_preds = []
            for seed in p["nn"]["seeds"]:
                model, d = train_nn(xtr, t_tr, e_tr, ids_tr, seed, p["nn"])
                seed_preds.append(predict_nn(model, xev))
                diag.append({"feature_set": fs, "seed": seed, **d})
            preds[("nn", fs)] = np.stack(seed_preds)
    return preds, diag


def c_of(pred, t, e):
    pred = np.atleast_2d(pred)
    return float(np.mean([concordance(t, e, r) for r in pred]))


def site_onehot(tss, train_mask, min_count=10):
    keep = tss[train_mask].value_counts()
    keep = keep[keep >= min_count].index
    cats = tss.where(tss.isin(keep), "other")
    x = pd.get_dummies(cats).astype(float)
    x = x.drop(columns=[c for c in ["other"] if c in x.columns])
    mean, sd = x[train_mask].mean(), x[train_mask].std().replace(0, 1)
    return ((x - mean) / sd).to_numpy(np.float32)


def dev(families):
    torch.set_num_threads(1)
    t0 = time.time()
    p, clin, reports, raw, labels, rows = load_inputs("dev")
    ids = labels.patient_id.to_numpy()
    fold_rows, nn_diag, site_rows, tss_rows = [], [], [], []
    for endpoint, (tc, ec) in ENDPOINTS.items():
        t_all, e_all = labels[tc].to_numpy(float), labels[ec].to_numpy(int)
        for rep in range(p["development"]["cv_repeats"]):
            folds = cv_folds(ids, labels.pfi_event, rep, p["development"]["cv_folds"])
            f = np.array([folds[i] for i in ids])
            for k in range(p["development"]["cv_folds"]):
                tr, ev = np.where(f != k)[0], np.where(f == k)[0]
                blocks, info = fit_transform_blocks(raw, rows[tr], rows[ev], p)
                b_tr = {m: v[0] for m, v in blocks.items()}
                b_ev = {m: v[1] for m, v in blocks.items()}
                preds, diag = fit_predict_all(p, b_tr, b_ev, t_all[tr], e_all[tr], ids[tr], families)
                for (fam, fs), pr in preds.items():
                    fold_rows.append({"endpoint": endpoint, "repeat": rep, "fold": k, "family": fam, "feature_set": fs, "c_index": c_of(pr, t_all[ev], e_all[ev]), "n_eval": len(ev), "events_eval": int(e_all[ev].sum()), **info})
                nn_diag += [{"endpoint": endpoint, "repeat": rep, "fold": k, **d} for d in diag]
                if endpoint == "PFI":
                    # Dev-only diagnostic 1: site-only Cox (how much can tissue-source-site alone rank PFI?)
                    tss = reports.tss.iloc[rows].reset_index(drop=True)
                    mask = np.zeros(len(rows), bool)
                    mask[tr] = True
                    xs = site_onehot(tss, mask)
                    cph = fit_cox(xs[tr], t_all[tr], e_all[tr], p["cox"]["penalizer"], 0.0)
                    site_rows.append({"repeat": rep, "fold": k, "c_index": concordance(t_all[ev], e_all[ev], predict_cox(cph, xs[ev]))})
                    # Dev-only diagnostic 2 (no outcomes): can text PCs identify the tissue source site?
                    if rep == 0:
                        tss_rows.append(tss_predictability(raw["text"][rows], tss.to_numpy(), tr, ev, p))
                print(f"[{time.time() - t0:6.0f}s] {endpoint} repeat {rep} fold {k} done", flush=True)
    df = pd.DataFrame(fold_rows)
    summary = []
    for (endpoint, fam, fs), g in df.groupby(["endpoint", "family", "feature_set"], sort=False):
        ref = df[(df.endpoint == endpoint) & (df.family == fam) & (df.feature_set == p["reference_feature_set"])].set_index(["repeat", "fold"]).c_index
        d = g.set_index(["repeat", "fold"]).c_index - ref
        summary.append({"endpoint": endpoint, "family": fam, "feature_set": fs, "c_index_mean": float(g.c_index.mean()), "c_index_sd_over_folds": float(g.c_index.std()), "delta_vs_clinical_mean": float(d.mean()), "delta_vs_clinical_sd": float(d.std()), "folds_delta_gt_0": int((d > 0).sum()), "n_folds": int(len(d))})
    nd = pd.DataFrame(nn_diag)
    out = {
        "phase": "development cross-validation (development split only; test labels never opened)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": f"{p['development']['cv_repeats']} x {p['development']['cv_folds']}-fold, stratified by PFI event, patient-level, all preprocessing refit inside each training fold",
        "n_dev": len(ids), "dev_events": {"PFI": int(labels.pfi_event.sum()), "OS": int(labels.os_event.sum())},
        "summary": summary,
        "folds": fold_rows,
        "nn_training": {"runs": len(nd), "hit_max_epochs": int(nd.hit_max_epochs.sum()) if len(nd) else 0, "median_best_epoch": float(nd.best_epoch.median()) if len(nd) else None},
        "diagnostics": {
            "site_only_cox_PFI": {"c_index_mean": float(np.mean([r["c_index"] for r in site_rows])), "c_index_sd": float(np.std([r["c_index"] for r in site_rows], ddof=1)), "n_folds": len(site_rows)},
            "text_to_tss": summarize_tss(tss_rows),
        },
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
        "environment": env_info(),
    }
    write_json(RESULTS / "dev_cv.json", out)
    print(pd.DataFrame(summary).round(3).to_string())


def tss_predictability(text, tss, tr, ev, p, min_count=15):
    from sklearn.linear_model import LogisticRegression

    counts = pd.Series(tss[tr]).value_counts()
    keep = set(counts[counts >= min_count].index)
    tr2 = [i for i in tr if tss[i] in keep]
    ev2 = [i for i in ev if tss[i] in keep]
    pca = BlockPCA(p["modalities"]["text"]["pca_components"]).fit(text[tr2])
    clf = LogisticRegression(max_iter=2000).fit(pca.transform(text[tr2]), tss[tr2])
    acc = float((clf.predict(pca.transform(text[ev2])) == tss[ev2]).mean())
    majority = float((tss[ev2] == counts[counts >= min_count].index[0]).mean())
    return {"accuracy": acc, "majority_baseline": majority, "n_sites": len(keep), "n_eval": len(ev2)}


def summarize_tss(rows):
    return {"accuracy_mean": float(np.mean([r["accuracy"] for r in rows])), "majority_baseline_mean": float(np.mean([r["majority_baseline"] for r in rows])), "n_sites": rows[0]["n_sites"], "folds": rows, "note": "Sites with >=15 training patients in the fold; text PCA-16 -> multinomial logistic regression; no outcomes used"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def frozen_hashes():
    h = {f: digest(ROOT / f) for f in FROZEN_FILES}
    h["data/embeddings/report_embeddings.npy"] = digest(EMB / "report_embeddings.npy")
    h.update({f"data/derived/{f}": digest(DERIVED / f) for f in ["clinical.csv", "reports.csv", "rna.npy", "dev_labels.csv", "test_labels.csv", "leakage_flags.csv"]})
    return h


def freeze():
    dirty = git("status", "--porcelain", "--", *FROZEN_FILES)
    if dirty:
        raise SystemExit(f"Commit frozen inputs first:\n{dirty}")
    write_json(RESULTS / "freeze.json", {"frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"), "hashes": frozen_hashes(), "dev_cv_sha256": digest(RESULTS / "dev_cv.json"), "statement": "Models, features, metrics, contrasts and decision rule are fixed in protocol.json at this commit. The test split is evaluated once after this file is committed."}, exclusive=True)
    print("Frozen at", git("rev-parse", "HEAD"))


def test():
    fz = json.loads((RESULTS / "freeze.json").read_text())
    if fz["hashes"] != frozen_hashes():
        raise SystemExit("Inputs differ from freeze.json; refusing to evaluate test set")
    if git("status", "--porcelain", "--", *FROZEN_FILES, "results/freeze.json"):
        raise SystemExit("Frozen files or freeze.json not committed")
    out_path = RESULTS / "test_results.json"
    if out_path.exists():
        raise SystemExit("Test evaluation already performed (one shot)")
    torch.set_num_threads(1)
    t0 = time.time()
    p, clin, reports, raw, dev_labels, dev_rows = load_inputs("dev")
    test_labels = pd.read_csv(DERIVED / "test_labels.csv")
    idx = pd.Series(range(len(clin)), index=clin.patient_id)
    test_rows = idx.loc[test_labels.patient_id].to_numpy()
    if not (clin.split.iloc[test_rows] == "test").all() or set(dev_rows) & set(test_rows):
        raise ValueError("Split integrity failure")
    flags = pd.read_csv(DERIVED / "leakage_flags.csv").set_index("patient_id").sensitivity_flag
    keep = ~flags.loc[test_labels.patient_id].to_numpy()

    blocks, info = fit_transform_blocks(raw, dev_rows, test_rows, p)
    b_tr = {m: v[0] for m, v in blocks.items()}
    b_te = {m: v[1] for m, v in blocks.items()}
    ref = p["reference_feature_set"]
    out = {"phase": "single held-out test evaluation", "created_at": datetime.now(timezone.utc).isoformat(), "freeze_commit": fz["git_commit"], "n_dev": len(dev_rows), "n_test": len(test_rows), "preprocessing": info, "endpoints": {}}
    for endpoint, (tc, ec) in ENDPOINTS.items():
        t_tr, e_tr = dev_labels[tc].to_numpy(float), dev_labels[ec].to_numpy(int)
        t_te, e_te = test_labels[tc].to_numpy(float), test_labels[ec].to_numpy(int)
        preds, diag = fit_predict_all(p, b_tr, b_te, t_tr, e_tr, dev_labels.patient_id.to_numpy())
        res = {"test_events": int(e_te.sum()), "families": {}}
        for fam in ["cox", "nn"]:
            fp = {fs: preds[(fam, fs)] for fs in p["feature_sets"]}
            contrasts = [(fs, ref) for fs in p["feature_sets"] if fs != ref] + [("clinical_rna_text", "clinical_rna")]
            rows, diffs, meta = paired_bootstrap(t_te, e_te, fp, contrasts, p["bootstrap"]["replicates"], p["bootstrap"]["seed"])
            for fs, pr in fp.items():
                rows[fs]["uno_c_tau1826"] = float(np.mean([uno_c(t_tr, e_tr, t_te, e_te, r, 1826) for r in np.atleast_2d(pr)]))
                if fam == "nn":
                    rows[fs]["c_index_per_seed"] = [concordance(t_te, e_te, r) for r in pr]
            s_rows, s_diffs, s_meta = paired_bootstrap(t_te[keep], e_te[keep], {fs: np.atleast_2d(pr)[:, keep] for fs, pr in fp.items()}, contrasts, p["bootstrap"]["replicates"], p["bootstrap"]["seed"])
            res["families"][fam] = {"models": rows, "contrasts": diffs, "bootstrap": meta,
                                    "sensitivity_excluding_flagged_reports": {"n": int(keep.sum()), "events": int(e_te[keep].sum()), "models": s_rows, "contrasts": s_diffs}}
            if fam == "nn":
                res["families"][fam]["training"] = diag
        out["endpoints"][endpoint] = res
    c = out["endpoints"]["PFI"]["families"]["cox"]["contrasts"]
    out["decision"] = {
        "rule": p["decision_rule"],
        "primary_contrast": {"name": "clinical_text - clinical", **c["clinical_text - clinical"]},
        "key_secondary_contrast": {"name": "clinical_rna_text - clinical_rna", **c["clinical_rna_text - clinical_rna"]},
        "text_adds_detectable_information": bool(c["clinical_text - clinical"]["ci95"][0] > 0),
        "text_adds_beyond_clinical_rna": bool(c["clinical_rna_text - clinical_rna"]["ci95"][0] > 0),
    }
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20, 1)
    out["environment"] = env_info()
    write_json(out_path, out, exclusive=True)
    print(json.dumps(out["decision"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dev", "freeze", "test"])
    ap.add_argument("--families", default="cox,nn")
    a = ap.parse_args()
    {"dev": lambda: dev(tuple(a.families.split(","))), "freeze": freeze, "test": test}[a.phase]()


if __name__ == "__main__":
    main()
