"""AHP eigenvector weights and consistency ratio."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Saaty RI for n=1..10
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: Sequence[Sequence[float]]) -> tuple[np.ndarray, float, float]:
    """Return (weights, lambda_max, CR) for a Saaty pairwise matrix."""
    A = np.asarray(pairwise, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("pairwise must be square")
    n = A.shape[0]
    # principal eigenvector (power iteration is enough for small n)
    w = np.ones(n, dtype=float) / n
    for _ in range(64):
        Aw = A @ w
        w_new = Aw / Aw.sum()
        if np.allclose(w, w_new, rtol=1e-9, atol=1e-12):
            w = w_new
            break
        w = w_new
    Aw = A @ w
    lambda_max = float((Aw / w).mean())
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RI.get(n, 1.49)
    cr = float(ci / ri) if ri > 0 else 0.0
    return w.astype(float), lambda_max, cr


def soft_score_from_result(
    result: dict,
    criterion_names: Sequence[str],
    w_c: np.ndarray,
) -> float:
    """Model soft probability from criterion scores or qualifies+confidence."""
    scores = result.get("scores") or {}
    vals = []
    for name in criterion_names:
        if name in scores and scores[name] is not None:
            try:
                vals.append(float(scores[name]))
            except (TypeError, ValueError):
                vals.append(None)
        else:
            vals.append(None)
    if all(v is not None for v in vals):
        s = np.asarray(vals, dtype=float)
        s = np.clip(s, 0.0, 1.0)
        return float(np.dot(w_c, s))
    # fallback
    conf = float(result.get("confidence") or 0.5)
    conf = min(max(conf, 0.0), 1.0)
    if bool(result.get("qualifies")):
        return conf
    return 1.0 - conf


def hard_label(p: float, *, hi: float = 0.55, lo: float = 0.45) -> str:
    if p >= hi:
        return "1"
    if p <= lo:
        return "0"
    return "uncertain"
