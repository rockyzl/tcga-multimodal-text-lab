"""Cox PH baseline, late-fusion Cox network, concordance and paired patient bootstrap."""
import hashlib
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn


# ---------------------------------------------------------------- Cox PH (lifelines)
def fit_cox(x, time, event, penalizer, l1_ratio):
    from lifelines import CoxPHFitter

    cols = [f"x{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    df["time"], df["event"] = np.asarray(time, float), np.asarray(event, int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio).fit(df, "time", "event")
    return cph


def predict_cox(cph, x):
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    return np.asarray(cph.predict_log_partial_hazard(df), dtype=float)


# ---------------------------------------------------------------- late-fusion network
class LateFusionCox(nn.Module):
    def __init__(self, dims, hidden=16, out=8, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out), nn.ReLU()) for d in dims)
        self.head = nn.Linear(out * len(dims), 1, bias=False)

    def forward(self, blocks):
        return self.head(torch.cat([b(x) for b, x in zip(self.branches, blocks)], 1)).squeeze(-1)


class BreslowLoss:
    """Negative Cox partial log-likelihood; each tied event uses the full risk set at that time."""

    def __init__(self, time, event):
        time, event = torch.as_tensor(time, dtype=torch.float32), torch.as_tensor(event, dtype=torch.float32)
        self.order = torch.argsort(time, descending=True, stable=True)
        t = time[self.order]
        self.event = event[self.order]
        _, counts = torch.unique_consecutive(t, return_counts=True)
        self.ends = torch.repeat_interleave(counts.cumsum(0) - 1, counts)
        if self.event.sum() <= 0:
            raise ValueError("Cox loss needs at least one observed event")

    def __call__(self, risk):
        r = risk[self.order]
        denom = torch.logcumsumexp(r, 0)[self.ends]
        return -((r - denom) * self.event).sum() / self.event.sum()


def inner_split(patient_ids, event, fraction, salt="inner-v1"):
    """Deterministic event-stratified inner validation split inside a training set (target read only to stratify)."""
    ids, event = list(patient_ids), np.asarray(event)
    val = np.zeros(len(ids), bool)
    for e in np.unique(event):
        idx = np.where(event == e)[0]
        order = sorted(idx, key=lambda i: hashlib.sha256(f"{ids[i]}|{salt}".encode()).hexdigest())
        val[order[: int(round(len(idx) * fraction))]] = True
    return ~val, val


def train_nn(blocks, time, event, patient_ids, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_m, val_m = inner_split(patient_ids, event, cfg["inner_validation_fraction"])
    tb = [torch.tensor(b[fit_m]) for b in blocks]
    vb = [torch.tensor(b[val_m]) for b in blocks]
    loss_fit = BreslowLoss(np.asarray(time)[fit_m], np.asarray(event)[fit_m])
    loss_val = BreslowLoss(np.asarray(time)[val_m], np.asarray(event)[val_m])
    model = LateFusionCox([b.shape[1] for b in blocks])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = loss_fit(model(tb))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_val(model(vb)))
        if v < best - cfg["min_delta"]:
            best, best_epoch, wait = v, epoch, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "stopped_epoch": epoch, "best_val_loss": best, "hit_max_epochs": epoch == cfg["max_epochs"] - 1}


def predict_nn(model, blocks):
    with torch.no_grad():
        return model([torch.tensor(b) for b in blocks]).numpy().astype(float)


# ---------------------------------------------------------------- evaluation
def pair_indices(time, event):
    """Comparable pairs (i, j): time_i < time_j strictly and event_i == 1."""
    time, event = np.asarray(time), np.asarray(event)
    return np.where((time[:, None] < time[None, :]) & (event[:, None] == 1))


def pair_scores(risk, left, right):
    r = np.asarray(risk, float)
    return (r[..., left] > r[..., right]) + 0.5 * (r[..., left] == r[..., right])


def concordance(time, event, risk):
    left, right = pair_indices(time, event)
    if not len(left):
        raise ValueError("No comparable survival pairs")
    return float(pair_scores(risk, left, right).mean())


def uno_c(train_time, train_event, time, event, risk, tau):
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv

    s_tr = Surv.from_arrays(np.asarray(train_event, bool), np.asarray(train_time, float))
    s_te = Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float))
    return float(concordance_index_ipcw(s_tr, s_te, np.asarray(risk, float), tau=tau)[0])


def paired_bootstrap(time, event, predictions, contrasts, replicates, seed):
    """predictions: name -> array (n,) or (seeds, n). Per-seed pairwise scores are averaged (no risk ensembling).
    Resamples patients with replacement; a pair is weighted by the product of its two patients' counts."""
    left, right = pair_indices(time, event)
    if not len(left):
        raise ValueError("No comparable survival pairs")
    scores = {m: np.atleast_2d(pair_scores(np.atleast_2d(p), left, right)).mean(0) for m, p in predictions.items()}
    rng = np.random.default_rng(seed)
    n = len(time)
    boot = {m: np.empty(replicates) for m in scores}
    for b in range(replicates):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        pw = w[left] * w[right]
        for m, s in scores.items():
            boot[m][b] = np.dot(pw, s) / pw.sum()
    rows = {m: {"c_index": float(s.mean()), "ci95": np.quantile(boot[m], [0.025, 0.975]).tolist()} for m, s in scores.items()}
    diffs = {}
    for a, ref in contrasts:
        d = boot[a] - boot[ref]
        diffs[f"{a} - {ref}"] = {"estimate": float(scores[a].mean() - scores[ref].mean()), "ci95": np.quantile(d, [0.025, 0.975]).tolist(), "fraction_boot_gt_0": float((d > 0).mean())}
    return rows, diffs, {"replicates": replicates, "seed": seed, "comparable_pairs": int(len(left))}
