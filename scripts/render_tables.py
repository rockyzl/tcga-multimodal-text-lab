"""Render Markdown result tables straight from results/*.json, so no number is typed by hand."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["clinical", "clinical_rna", "clinical_text", "text", "clinical_rna_text", "rna"]


def f3(x):
    return f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def dev_table(dev, endpoint):
    rows = {(r["family"], r["feature_set"]): r for r in dev["summary"] if r["endpoint"] == endpoint}
    out = [f"| Feature set | Cox C (mean ± SD, 15 folds) | Cox Δ vs clinical | NN C (mean ± SD) | NN Δ vs clinical |", "|---|---|---|---|---|"]
    for fs in ORDER:
        cells = [fs]
        for fam in ["cox", "nn"]:
            r = rows.get((fam, fs))
            if r is None:
                cells += ["—", "—"]
                continue
            cells.append(f"{f3(r['c_index_mean'])} ± {f3(r['c_index_sd_over_folds'])}")
            cells.append("ref" if fs == "clinical" else f"{r['delta_vs_clinical_mean']:+.3f} ± {f3(r['delta_vs_clinical_sd'])} ({r['folds_delta_gt_0']}/{r['n_folds']} folds > 0)")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def test_table(res, endpoint, family, key="models", contrasts_key="contrasts", sens=False):
    fam = res["endpoints"][endpoint]["families"][family]
    block = fam["sensitivity_excluding_flagged_reports"] if sens else fam
    models, contrasts = block["models"], block["contrasts"]
    head = "| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] |" + ("" if sens else " Uno C (τ=5y) |")
    out = [head, "|---|---|---|" + ("" if sens else "---|")]
    for fs in ORDER:
        m = models[fs]
        d = "ref" if fs == "clinical" else f"{contrasts[f'{fs} - clinical']['estimate']:+.3f} {ci(contrasts[f'{fs} - clinical']['ci95'])}"
        row = f"| {fs} | {f3(m['c_index'])} {ci(m['ci95'])} | {d} |"
        if not sens:
            row += f" {f3(m['uno_c_tau1826'])} |"
        out.append(row)
    k = contrasts["clinical_rna_text - clinical_rna"]
    out.append(f"\nclinical_rna_text − clinical_rna: {k['estimate']:+.3f} {ci(k['ci95'])}")
    return "\n".join(out)


def main():
    dev = json.loads((ROOT / "results/dev_cv.json").read_text())
    parts = ["## Development CV (dev split only)"]
    for ep in ["PFI", "OS"]:
        parts += [f"\n### {ep}\n", dev_table(dev, ep)]
    d = dev["diagnostics"]
    parts.append(f"\nSite-only Cox (PFI, dev CV): C = {f3(d['site_only_cox_PFI']['c_index_mean'])} ± {f3(d['site_only_cox_PFI']['c_index_sd'])}")
    parts.append(f"Text PCs → tissue source site: accuracy {f3(d['text_to_tss']['accuracy_mean'])} vs majority baseline {f3(d['text_to_tss']['majority_baseline_mean'])} ({d['text_to_tss']['n_sites']} sites)")
    tp = ROOT / "results/test_results.json"
    if tp.exists():
        res = json.loads(tp.read_text())
        for ep in ["PFI", "OS"]:
            e = res["endpoints"][ep]
            for fam in ["cox", "nn"]:
                parts += [f"\n## Test — {ep} — {fam} (n={res['n_test']}, events={e['test_events']})\n", test_table(res, ep, fam)]
            s = e["families"]["cox"]["sensitivity_excluding_flagged_reports"]
            parts += [f"\n### Sensitivity — {ep} — cox, flagged reports removed (n={s['n']}, events={s['events']})\n", test_table(res, ep, "cox", sens=True)]
        parts.append("\n## Decision\n\n```json\n" + json.dumps(res["decision"], indent=2) + "\n```")
    text = "\n".join(parts)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
