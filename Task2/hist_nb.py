"""Histogram naive Bayes for synthetic tabular tasks whose features are independent given the class.
Each numeric feature: class-conditional density on quantile bins (fit on the training fold only), NaN is its own bin.
Categorical / low-cardinality features: smoothed class-conditional frequencies (NaN its own level)."""
import numpy as np, pandas as pd


class HistNB:
    def __init__(self, n_bins=40, alpha=1.0, skip=()):
        self.n_bins, self.alpha, self.skip = n_bins, alpha, set(skip)

    def _codes(self, s, c):
        if c in self.edges:
            e = self.edges[c]
            v = s.to_numpy(float)
            code = np.clip(np.searchsorted(e, v, side='right') - 1, 0, len(e) - 2)
            return np.where(np.isnan(v), len(e) - 1, code)
        lv = self.levels[c]
        m = {k: i for i, k in enumerate(lv)}
        return np.array([m.get(x, len(lv)) if not (isinstance(x, float) and np.isnan(x)) else len(lv) + 1 for x in s.astype(object)])

    def fit(self, X, y):
        self.classes = np.unique(y); K = len(self.classes)
        self.edges, self.levels, self.logp = {}, {}, {}
        self.cols = [c for c in X.columns if c not in self.skip]
        for c in self.cols:
            s = X[c]
            if s.dtype.kind in 'fi' and s.nunique() > 30:
                e = np.unique(np.nanquantile(s.to_numpy(float), np.linspace(0, 1, self.n_bins + 1)))
                e[0], e[-1] = -np.inf, np.inf
                self.edges[c] = e; nb = len(e)            # len(e)-1 bins + NaN bin
            else:
                self.levels[c] = sorted(s.dropna().unique().tolist()); nb = len(self.levels[c]) + 2
            code = self._codes(s, c)
            cnt = np.zeros((K, nb))
            for k, cl in enumerate(self.classes):
                cnt[k] = np.bincount(code[y == cl], minlength=nb)[:nb]
            self.logp[c] = np.log((cnt + self.alpha) / (cnt.sum(1, keepdims=True) + self.alpha * nb))
        self.prior = np.log(np.array([(y == cl).mean() for cl in self.classes]))
        return self

    def log_lik(self, X):
        L = np.zeros((len(X), len(self.classes)))
        for c in self.cols:
            L += self.logp[c][:, self._codes(X[c], c)].T
        return L

    def predict_proba(self, X):
        L = self.log_lik(X) + self.prior
        L -= L.max(1, keepdims=True); P = np.exp(L)
        return P / P.sum(1, keepdims=True)
