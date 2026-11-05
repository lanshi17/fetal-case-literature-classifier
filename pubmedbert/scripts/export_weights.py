"""Export the trained classification head (embedding + logistic regression, no Claude stage) as plain numpy/JSON files."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

out = Path("out")
dest = Path("weights")
dest.mkdir(exist_ok=True)
head = joblib.load(out / "head_v3.joblib")
coef, icpt = head.coef_.ravel().astype(np.float64), float(head.intercept_[0])
np.savez(dest / "head_v3.npz", coef=coef, intercept=np.array([icpt]))
joblib.dump(head, dest / "head_v3.joblib")
met = json.loads((out / "metrics_v3.json").read_text())
cfg = {
    "name": "casefilter-head-v3",
    "task": "P(paper has >=1 fetal case with prenatal ultrasound abnormality + one of 11 genetic tests)",
    "embedding_model": "NeuML/pubmedbert-base-embeddings",
    "chunk_tokens": 480, "max_chunks": 12, "keep_first_chunks": 2,
    "pooling": "concat[mean(all chunks) , mean(first 2 chunks)], each L2-normalised -> 1536 dims",
    "head": {"type": "logistic_regression", "C": met["C"], "n_features": int(coef.size), "file": "head_v3.npz"},
    "threshold": 0.5,
    "trained_on": {"n_train_for_validation_split": met["n_train"], "final_fit": "all 965 Claude-labelled + curated gold papers"},
    "validation_at_0.5": {k: met["val_at_0_5"][k] for k in ("n", "auc", "precision", "recall", "specificity", "f1", "balanced_acc")},
    "caveat": "validation labels came from Claude, not humans; metrics measure agreement with Claude",
    "sha256_npz": hashlib.sha256((dest / "head_v3.npz").read_bytes()).hexdigest(),
}
(dest / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print("features", coef.size, "intercept", round(icpt, 4), "->", dest.resolve())
