"""Training-only preprocessing for each modality. Every statistic is fit on the rows passed to fit()."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ClinicalPreprocessor:
    """Median imputation + fixed missing indicators + standardization, all from training rows."""

    def __init__(self, missing_indicator_cols):
        self.missing_indicator_cols = list(missing_indicator_cols)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.median = np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0)
        filled = self._raw(x)
        self.mean = filled.mean(0)
        self.scale = filled.std(0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def _raw(self, x):
        filled = np.where(np.isnan(x), self.median, x)
        ind = np.isnan(x[:, self.missing_indicator_cols]).astype(float)
        return np.concatenate([filled, ind], axis=1)

    def transform(self, x):
        out = (self._raw(np.asarray(x, dtype=float)) - self.mean) / self.scale
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite clinical features")
        return out.astype(np.float32)


class BlockPCA:
    """Optional top-variance column filter -> standardize -> PCA -> standardize scores. Fit on training rows only."""

    def __init__(self, n_components, top_variance=None):
        self.n_components = n_components
        self.top_variance = top_variance

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite input to BlockPCA")
        var = x.var(0)
        keep = np.where(var > 1e-8)[0]
        if self.top_variance is not None and len(keep) > self.top_variance:
            keep = keep[np.argsort(-var[keep], kind="stable")[: self.top_variance]]
        self.columns = np.sort(keep)
        z = x[:, self.columns]
        self.pre = StandardScaler().fit(z)
        self.pca = PCA(n_components=self.n_components, svd_solver="full", random_state=0).fit(self.pre.transform(z))
        self.post = StandardScaler().fit(self.pca.transform(self.pre.transform(z)))
        self.explained_variance_ratio = float(self.pca.explained_variance_ratio_.sum())
        return self

    def transform(self, x):
        z = np.asarray(x, dtype=np.float32)[:, self.columns]
        return self.post.transform(self.pca.transform(self.pre.transform(z))).astype(np.float32)


def fit_transform_blocks(raw, train_idx, eval_idx, protocol):
    """raw: dict modality -> full matrix (rows aligned to one patient order).
    Returns dict modality -> (train_matrix, eval_matrix) and fit diagnostics. Only train_idx rows touch fit()."""
    train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
    if len(np.intersect1d(train_idx, eval_idx)):
        raise ValueError("Train and evaluation rows overlap")
    out, info = {}, {}
    for name, x in raw.items():
        if name == "clinical":
            idx = [protocol["clinical_features"].index(c) for c in protocol["missing_indicator_features"]]
            p = ClinicalPreprocessor(idx).fit(x[train_idx])
        else:
            cfg = protocol["modalities"][name]
            p = BlockPCA(cfg["pca_components"], cfg.get("top_variance_genes")).fit(x[train_idx])
            info[f"{name}_pca_explained_variance"] = p.explained_variance_ratio
        out[name] = (p.transform(x[train_idx]), p.transform(x[eval_idx]))
    return out, info
