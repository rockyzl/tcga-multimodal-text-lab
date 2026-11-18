"""Frozen biomedical transformer embeddings for pathology reports, plus text-only audits.

Nothing in this module reads outcome labels.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DERIVED, ROOT, digest

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MODEL_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
MODEL_LICENSE = "MIT (Hugging Face model card tag license:mit)"
MAX_LENGTH = 512
STRIDE = 64
BATCH_CHUNKS = 8
EMB = ROOT / "data" / "embeddings"

# Text-only outcome-language audit. Patterns are fixed here before any model is evaluated.
LEAKAGE_PATTERNS = {
    "death_or_autopsy": r"\b(autops\w*|expired|deceased|died|death|post[- ]?mortem)\b",
    "relapse_or_progression_of_disease": r"\b(relaps\w*|disease progression|progressive disease|progressed)\b",
    "recurrence_score_assay": r"(oncotype|recurrence scor|rate of distant recurrence|estimated recurrence)",
    "recurrent_or_prior_breast_cancer": r"(\brecurrent\b.{0,40}(carcinoma|cancer|tumou?r)|history of.{0,30}(breast )?(carcinoma|cancer)|prior.{0,20}(breast )?(carcinoma|cancer))",
    "neoadjuvant_or_post_treatment": r"(neoadjuvant|neo-adjuvant|post[- ]?(chemo\w*|treatment|therapy)|treatment effect|therapy effect)",
    "follow_up_word": r"follow[- ]?up",
    "survival_word": r"\bsurviv\w*",
    "distant_metastasis_named_site": r"metasta\w*.{0,40}\b(liver|lung|bone|brain)\b|\b(liver|lung|bone|brain)\b.{0,40}metasta",
}
# Flags that define the prespecified sensitivity subset (excluded from the sensitivity evaluation).
SENSITIVITY_FLAGS = ["death_or_autopsy", "relapse_or_progression_of_disease", "recurrence_score_assay", "recurrent_or_prior_breast_cancer", "neoadjuvant_or_post_treatment"]


def leakage_flags(texts):
    low = pd.Series(texts).fillna("").str.lower()
    return pd.DataFrame({k: low.map(lambda s, p=p: bool(re.search(p, s))) for k, p in LEAKAGE_PATTERNS.items()})


def leakage_scan():
    reports = pd.read_csv(DERIVED / "reports.csv")
    flags = leakage_flags(reports.text)
    examples = {}
    for k, p in LEAKAGE_PATTERNS.items():
        rows = []
        for s in reports.text[flags[k]].str.lower().head(3):
            m = re.search(p, s)
            rows.append("..." + s[max(0, m.start() - 80): m.end() + 80].replace("\n", " ") + "...")
        examples[k] = rows
    flagged = flags[SENSITIVITY_FLAGS].any(axis=1)
    out = {
        "n_reports": len(reports),
        "pattern_counts": flags.sum().astype(int).to_dict(),
        "patterns": LEAKAGE_PATTERNS,
        "sensitivity_flags": SENSITIVITY_FLAGS,
        "sensitivity_flagged_total": int(flagged.sum()),
        "sensitivity_flagged_by_split": reports.assign(f=flagged.values).groupby("split").f.sum().astype(int).to_dict(),
        "examples": examples,
        "reads_outcomes": False,
    }
    pd.DataFrame({"patient_id": reports.patient_id, "sensitivity_flag": flagged.values}).to_csv(DERIVED / "leakage_flags.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "leakage_scan.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["pattern_counts", "sensitivity_flagged_total", "sensitivity_flagged_by_split"]}, indent=2))


def chunk_report(tokenizer, text):
    """Split one report into <=512-token windows (each with its own [CLS]/[SEP]) overlapping by STRIDE tokens."""
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, stride=STRIDE, return_overflowing_tokens=True, padding=False)
    return enc["input_ids"]


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


def embed(threads=6):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    EMB.mkdir(parents=True, exist_ok=True)
    reports = pd.read_csv(DERIVED / "reports.csv")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
    partial = EMB / "partial.npz"
    done = dict(np.load(partial, allow_pickle=False)) if partial.exists() else {}
    vectors = {k: v for k, v in done.items()}
    chunk_counts, token_counts = {}, {}
    t0 = time.time()
    for i, (pid, text) in enumerate(zip(reports.patient_id, reports.text.fillna(""))):
        ids = chunk_report(tok, text)
        chunk_counts[pid] = len(ids)
        token_counts[pid] = int(len(tok(text, add_special_tokens=False)["input_ids"]))
        if pid in vectors:
            continue
        chunk_vecs = []
        with torch.inference_mode():
            for b in range(0, len(ids), BATCH_CHUNKS):
                batch = tok.pad({"input_ids": ids[b:b + BATCH_CHUNKS]}, return_tensors="pt")
                hidden = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                chunk_vecs.append(mean_pool(hidden, batch["attention_mask"]))
        vectors[pid] = torch.cat(chunk_vecs).mean(0).numpy().astype(np.float32)
        if (i + 1) % 50 == 0:
            np.savez(partial, **vectors)
            print(f"{i + 1}/{len(reports)} reports, {time.time() - t0:.0f}s", flush=True)
    matrix = np.stack([vectors[p] for p in reports.patient_id])
    np.save(EMB / "report_embeddings.npy", matrix)
    meta = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "frozen": True, "fine_tuned": False,
        "max_length": MAX_LENGTH, "stride": STRIDE, "pooling": "attention-masked mean of last hidden layer per chunk (includes [CLS]/[SEP]); unweighted mean over chunks",
        "n_reports": len(reports), "dim": int(matrix.shape[1]), "patient_order_sha256": __import__("hashlib").sha256("\n".join(reports.patient_id).encode()).hexdigest(),
        "chunks_per_report": pd.Series(chunk_counts).describe().round(2).to_dict(),
        "tokens_per_report": pd.Series(token_counts).describe().round(1).to_dict(),
        "reports_needing_more_than_one_chunk": int((pd.Series(chunk_counts) > 1).sum()),
        "seconds": round(time.time() - t0, 1), "torch_threads": threads,
        "embeddings_sha256": digest(EMB / "report_embeddings.npy"),
    }
    (EMB / "embedding_meta.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "text_embedding.json").write_text(json.dumps(meta, indent=2))
    partial.unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["scan", "embed"])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    leakage_scan() if a.step == "scan" else embed(a.threads)


if __name__ == "__main__":
    main()
