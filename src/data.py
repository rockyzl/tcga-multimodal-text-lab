"""Download public TCGA-BRCA sources, record provenance, join at patient level, assign splits.

Outcome columns are read once, here, only to (a) apply prespecified eligibility rules and
(b) stratify the patient-level split by PFI event status. No model sees them in this step.
"""
import argparse
import gzip
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

SOURCES = {
    "reports": {
        "file": "TCGA_Reports.csv.zip",
        "url": "https://raw.githubusercontent.com/jkefeli/tcga-path-reports/main/TCGA_Reports.csv.zip",
        "license": "CC BY 4.0 (Mendeley Data DOI 10.17632/hyg5xkznpx.1); code repository jkefeli/tcga-path-reports is MIT",
        "citation": "Kefeli J, Tatonetti N. TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models. Patterns 2024.",
    },
    "report_cancer_type": {
        "file": "tcga_patient_to_cancer_type.csv",
        "url": "https://raw.githubusercontent.com/jkefeli/tcga-path-reports/main/data/tcga_metadata/tcga_patient_to_cancer_type.csv",
        "license": "MIT (jkefeli/tcga-path-reports)",
        "citation": "Kefeli & Tatonetti 2024 repository metadata (cross-check only; CDR type is authoritative)",
    },
    "cdr": {
        "file": "TCGA-CDR-SupplementalTableS1.xlsx",
        "url": "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81",
        "license": "NCI Genomic Data Commons open-access PanCanAtlas publication supplement; TCGA open-access data (NIH TCGA publication guidelines, no access restrictions)",
        "citation": "Liu J, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell 2018;173:400-416.",
    },
    "rna": {
        "file": "HiSeqV2.gz",
        "url": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
        "license": "UCSC Xena TCGA hub; dataset metadata security=public, redistribution=true; underlying TCGA open-access Level 3 data",
        "citation": "Goldman MJ, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol 2020. Data: TCGA BRCA IlluminaHiSeq RNASeqV2, log2(RSEM norm_count+1), Xena version 2017-10-13.",
    },
    "rna_metadata": {
        "file": "HiSeqV2.json",
        "url": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.json",
        "license": "UCSC Xena dataset metadata",
        "citation": "UCSC Xena",
    },
}

# Columns that are follow-up derived, endpoint derived, or track follow-up length.
FORBIDDEN_PREDICTORS = [
    "vital_status", "tumor_status", "last_contact_days_to", "death_days_to", "cause_of_death",
    "new_tumor_event_type", "new_tumor_event_site", "new_tumor_event_site_other",
    "new_tumor_event_dx_days_to", "treatment_outcome_first_course", "margin_status", "residual_tumor",
    "birth_days_to", "initial_pathologic_dx_year", "Redaction",
    "OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time",
]
CLINICAL_RAW = ["age_at_initial_pathologic_diagnosis", "ajcc_pathologic_tumor_stage", "histological_type"]
CLINICAL_FEATURES = ["age", "stage", "histology_lobular", "histology_other"]
SPLIT_SALT = "tcga-text-v1"
TEST_FRACTION = 0.25


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for key, s in SOURCES.items():
        path = RAW / s["file"]
        receipt_path = RAW / f"{s['file']}.receipt.json"
        if force or not path.exists():
            with requests.get(s["url"], stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            receipt_path.write_text(json.dumps({"url": s["url"], "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": digest(path), "bytes": path.stat().st_size}, indent=2))
        if not receipt_path.exists():  # file fetched manually before this script existed
            receipt_path.write_text(json.dumps({"url": s["url"], "retrieved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "sha256": digest(path), "bytes": path.stat().st_size}, indent=2))
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != digest(path):
            raise ValueError(f"Raw file changed since receipt: {path}")
        receipts[key] = {**receipt, "file": s["file"], "license": s["license"], "citation": s["citation"]}
    return receipts


def patient_from_report(filename):
    pid = filename.split(".")[0]
    if not (len(pid) == 12 and pid.startswith("TCGA-")):
        raise ValueError(f"Unexpected report filename: {filename}")
    return pid


def stage_ordinal(value):
    """AJCC pathologic stage collapsed to I-IV. Stage X, discrepancy and not-available are missing."""
    if not isinstance(value, str) or not value.startswith("Stage "):
        return np.nan
    roman = value.replace("Stage ", "").rstrip("ABC")
    return {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}.get(roman, np.nan)


def clinical_features(cdr):
    out = pd.DataFrame({"patient_id": cdr.bcr_patient_barcode.values})
    out["age"] = pd.to_numeric(cdr.age_at_initial_pathologic_diagnosis, errors="coerce").values
    out["stage"] = cdr.ajcc_pathologic_tumor_stage.map(stage_ordinal).values
    hist = cdr.histological_type.fillna("")
    out["histology_lobular"] = (hist == "Infiltrating Lobular Carcinoma").astype(float).values
    out["histology_other"] = (~hist.isin(["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma"])).astype(float).values
    return out


def load_cdr_brca():
    cdr = pd.read_excel(RAW / SOURCES["cdr"]["file"], sheet_name="TCGA-CDR")
    return cdr[cdr["type"] == "BRCA"].reset_index(drop=True)


def load_reports_brca(brca_ids):
    with zipfile.ZipFile(RAW / SOURCES["reports"]["file"]) as z:
        reports = pd.read_csv(z.open("TCGA_Reports.csv"))
    reports["patient_id"] = reports.patient_filename.map(patient_from_report)
    if reports.patient_id.duplicated().any():
        raise ValueError("More than one report per patient; prespecify a rule first")
    return reports[reports.patient_id.isin(brca_ids)].reset_index(drop=True), len(reports)


def rna_primary_tumor_columns():
    """Return (sample_columns, patient_ids) for primary-tumor (-01) samples, one per patient."""
    with gzip.open(RAW / SOURCES["rna"]["file"], "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    samples = pd.Series(header[1:])
    primary = samples[samples.str[13:15] == "01"]
    patients = primary.str[:12]
    if patients.duplicated().any():
        raise ValueError("More than one primary-tumor RNA sample per patient; prespecify a rule first")
    return header, primary.tolist(), patients.tolist()


def load_rna(patient_ids):
    """Read the matrix once, keep only needed primary-tumor columns as float32."""
    header, samples, patients = rna_primary_tumor_columns()
    sample_for = dict(zip(patients, samples))
    wanted = [sample_for[p] for p in patient_ids]
    df = pd.read_csv(RAW / SOURCES["rna"]["file"], sep="\t", usecols=[header[0], *wanted], index_col=0, dtype={w: np.float32 for w in wanted})
    df = df[wanted]
    df = df[df.notna().all(axis=1)]
    matrix = df.T.to_numpy(dtype=np.float32)
    return matrix, df.index.tolist(), wanted


def split_for(patient_ids, events, salt=SPLIT_SALT, test_fraction=TEST_FRACTION):
    """Deterministic patient-level split stratified by event: rank by salted hash within each stratum."""
    df = pd.DataFrame({"patient_id": list(patient_ids), "event": list(events)})
    if df.patient_id.duplicated().any():
        raise ValueError("Duplicate patient in split input")
    df["h"] = df.patient_id.map(lambda p: hashlib.sha256(f"{p}|{salt}".encode()).hexdigest())
    out = {}
    for _, g in df.groupby("event"):
        g = g.sort_values("h")
        n_test = int(round(len(g) * test_fraction))
        for i, pid in enumerate(g.patient_id):
            out[pid] = "test" if i < n_test else "dev"
    return out


def cv_folds(patient_ids, events, repeat, n_folds=5, salt=SPLIT_SALT):
    """Stratified fold labels for development patients; each repeat uses a different salt."""
    df = pd.DataFrame({"patient_id": list(patient_ids), "event": list(events)})
    df["h"] = df.patient_id.map(lambda p: hashlib.sha256(f"{p}|{salt}|cv{repeat}".encode()).hexdigest())
    out = {}
    for _, g in df.groupby("event"):
        for i, pid in enumerate(g.sort_values("h").patient_id):
            out[pid] = i % n_folds
    return out


def build(receipts):
    DERIVED.mkdir(parents=True, exist_ok=True)
    steps = []
    cdr = load_cdr_brca()
    steps.append({"step": "TCGA-CDR rows with type == BRCA", "n": len(cdr)})
    if cdr.bcr_patient_barcode.duplicated().any():
        raise ValueError("Duplicate CDR barcode")
    cdr = cdr[cdr.Redaction.isna()]
    steps.append({"step": "exclude CDR Redaction == 'Redacted'", "n": len(cdr)})
    ok = cdr["PFI.time"].gt(0) & cdr["PFI"].isin([0, 1]) & cdr["OS.time"].gt(0) & cdr["OS"].isin([0, 1])
    cdr = cdr[ok].reset_index(drop=True)
    steps.append({"step": "require PFI.time > 0 and OS.time > 0 with known event status", "n": len(cdr)})

    reports, n_reports_all = load_reports_brca(set(cdr.bcr_patient_barcode))
    _, _, rna_patients = rna_primary_tumor_columns()
    type_map = pd.read_csv(RAW / SOURCES["report_cancer_type"]["file"])
    repo_brca = set(type_map[type_map.cancer_type == "BRCA"].patient_id)

    eligible = set(cdr.bcr_patient_barcode)
    has_text = set(reports.patient_id)
    has_rna = set(rna_patients) & eligible
    modality_n = {
        "clinical_eligible": len(eligible),
        "clinical_with_report": len(eligible & has_text),
        "clinical_with_rna_primary_tumor": len(has_rna),
        "all_three": len(eligible & has_text & has_rna),
        "cancer_type_map_brca_patients": len(repo_brca),
        "cancer_type_map_brca_not_in_cdr_eligible": len(repo_brca - eligible),
        "reports_total_all_cancers": n_reports_all,
    }
    joint = sorted(eligible & has_text & has_rna)
    steps.append({"step": "has pathology report (TCGA-Reports)", "n": len(eligible & has_text)})
    steps.append({"step": "and has primary-tumor RNA-seq (Xena HiSeqV2, sample type 01)", "n": len(joint)})

    cdr = cdr.set_index("bcr_patient_barcode").loc[joint].reset_index()
    clin = clinical_features(cdr)
    labels = pd.DataFrame({"patient_id": joint, "pfi_time": cdr["PFI.time"].values, "pfi_event": cdr["PFI"].astype(int).values, "os_time": cdr["OS.time"].values, "os_event": cdr["OS"].astype(int).values})
    text = reports.set_index("patient_id").loc[joint].reset_index()[["patient_id", "patient_filename", "text"]]
    for a, b in [(clin, labels), (clin, text)]:
        if a.patient_id.tolist() != b.patient_id.tolist():
            raise ValueError("Join order mismatch")

    rna, genes, samples = load_rna(joint)
    rna_ids = [s[:12] for s in samples]
    if rna_ids != joint:
        raise ValueError("RNA patient order mismatch")

    membership = split_for(joint, labels.pfi_event)
    clin["split"] = clin.patient_id.map(membership)
    labels["split"] = labels.patient_id.map(membership)
    text["split"] = text.patient_id.map(membership)
    text["tss"] = text.patient_id.str[5:7]

    clin.to_csv(DERIVED / "clinical.csv", index=False)
    text.to_csv(DERIVED / "reports.csv", index=False)
    for part in ["dev", "test"]:
        labels[labels.split == part].to_csv(DERIVED / f"{part}_labels.csv", index=False)
    np.save(DERIVED / "rna.npy", rna)
    (DERIVED / "rna_genes.json").write_text(json.dumps({"genes": genes, "samples": samples}))
    (ROOT / "data" / "membership.json").write_text(json.dumps(membership, sort_keys=True, indent=1))

    split_summary = {part: {"n": int((labels.split == part).sum()), "pfi_events": int(labels[labels.split == part].pfi_event.sum()), "os_events": int(labels[labels.split == part].os_event.sum())} for part in ["dev", "test"]}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_steps": steps,
        "modality_counts": modality_n,
        "joint_cohort": len(joint),
        "split": split_summary,
        "split_rule": f"Within each PFI-event stratum, sort patients by SHA256(patient_id|{SPLIT_SALT}); first round({TEST_FRACTION}*n) are test. Event status is read only for stratification.",
        "rna_genes_without_missing": len(genes),
        "clinical_missing": clin[CLINICAL_FEATURES].isna().sum().astype(int).to_dict(),
        "stage_raw_counts_joint": cdr.ajcc_pathologic_tumor_stage.fillna("NaN").value_counts().to_dict(),
        "histology_raw_counts_joint": cdr.histological_type.fillna("NaN").value_counts().to_dict(),
        "sources": receipts,
        "derived_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(DERIVED.glob("*")) if p.is_file() and p.suffix in {".csv", ".npy", ".json"}},
        "membership_sha256": digest(ROOT / "data" / "membership.json"),
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["cohort_steps", "modality_counts", "split", "clinical_missing"]}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()
    build(download(args.force_download))


if __name__ == "__main__":
    main()
