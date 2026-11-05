"""Shared screening criteria and prompt schema for multi-model judges."""

from __future__ import annotations

import json
import re
from typing import Any

CRITERIA = """You screen papers for a prenatal genetic-variant case database.
A paper QUALIFIES if it reports at least ONE individual human fetus (or pregnancy) such that ALL hold:
1. FETAL: human fetus / prenatal / pregnancy context (not only postnatal child unless clearly the same prenatal case).
2. PHENO_US: a prenatal ultrasound phenotype abnormality is described (structural anomaly or soft marker, e.g. ventriculomegaly, increased NT, cardiac defect).
3. GENETIC_TEST: at least one of these tests was performed and linked to that fetus (positive or negative): Karyotyping, CMA, FISH, CNV-seq, WES, Targeted Panel, MLPA, Sanger, WGS, PCR, mtDNA-seq.
4. CASE_PRIMARY: primary research or case report/series (NOT narrative/systematic review, meta-analysis, guideline, animal/cell-only, or pure method paper). Cohort studies qualify only if ultrasound findings and test results of studied fetuses are described (not just aggregate screening accuracy).
When genuinely ambiguous, prefer qualifies=true with lower confidence (recall over precision)."""

SCHEMA_HINT = (
    "Reply with ONLY one JSON object (no markdown): "
    '{"qualifies": bool, "confidence": 0-1, '
    '"scores": {"FETAL": 0-1, "PHENO_US": 0-1, "GENETIC_TEST": 0-1, "CASE_PRIMARY": 0-1}, '
    '"methods": [..from the 11 names..], '
    '"case_type": "case"|"cohort"|"none", '
    '"reason": "<=40 words"}'
)

CRITERION_NAMES = ("FETAL", "PHENO_US", "GENETIC_TEST", "CASE_PRIMARY")

# Default Saaty pairwise matrix for criteria (order = CRITERION_NAMES).
# GENETIC_TEST and PHENO_US slightly preferred over CASE_PRIMARY; FETAL mid.
# Row i vs col j: how much more important is i than j.
DEFAULT_CRITERION_PAIRWISE = [
    # FETAL, PHENO, GENETIC, CASE
    [1, 1 / 2, 1 / 3, 2],
    [2, 1, 1, 3],
    [3, 1, 1, 3],
    [1 / 2, 1 / 3, 1 / 3, 1],
]


def build_prompt(title: str, text: str, *, max_chars: int = 16000, pub_types: str = "") -> str:
    body = (text or "")[:max_chars]
    return (
        f"{CRITERIA}\n\n{SCHEMA_HINT}\n\nTITLE: {title or ''}\n"
        f"PUBLICATION TYPES: {pub_types or ''}\nTEXT:\n{body}"
    )


def parse_reply(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "qualifies" not in obj:
        return None
    return obj
