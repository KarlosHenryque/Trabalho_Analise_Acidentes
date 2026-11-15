"""Testes de associação, medidas de variabilidade e detecção de outliers (EDA)."""
import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(a, b):
    """Qui-quadrado, p-valor e V de Cramér (com correção de Bergsma) entre duas categóricas."""
    t = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(t)
    n = t.values.sum()
    r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return chi2, p, np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1)))


def variabilidade(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Medidas de posição e dispersão por coluna numérica."""
    rows = {}
    for c in cols:
        s = df[c].dropna()
        rows[c] = {"n": len(s), "média": s.mean(), "mediana": s.median(), "desvio": s.std(),
                   "variância": s.var(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
                   "amplitude": s.max() - s.min(), "CV (%)": 100 * s.std() / s.mean() if s.mean() else np.nan,
                   "assimetria": s.skew()}
    return pd.DataFrame(rows).T


def outliers_iqr(s: pd.Series, k=1.5) -> dict:
    """Cercas de Tukey: [Q1 - k*IQR, Q3 + k*IQR]."""
    s = s.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    n = int(((s < lo) | (s > hi)).sum())
    return {"Q1": q1, "Q3": q3, "limite_inf": lo, "limite_sup": hi, "n_outliers": n, "% outliers": 100 * n / len(s)}
