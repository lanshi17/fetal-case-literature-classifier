#!/usr/bin/env python3
"""Post-model rule layer for hard-negative exclusion (recall-preserving).

Design
------
Model score → provisional decision (relevant / review / irrelevant), then rules:

1. **force_irrelevant** — high-precision patterns for NIPT/screening cohorts,
   reviews/guidelines, databases, pure prediction/GWAS. Blocked if a *strong
   case-report* cue is present (protect true positives).
2. **demote_to_review** — medium-confidence cohort/screening cues: never flip
   relevant → irrelevant directly; send to human review instead.
3. Rules never promote irrelevant → relevant (model owns recall).

Usage
-----
    from rule_layer import apply_rules
    out = apply_rules(title, abstract, text, score=0.9, threshold=0.76)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

REVIEW_MARGIN = 0.15

# --- hard exclude (high precision) ------------------------------------------
RX_REVIEW = re.compile(
    r"(?i)\b("
    r"systematic\s+review|literature\s+review|narrative\s+review|"
    r"clinical\s+review|scoping\s+review|meta[- ]analysis|"
    r"practice\s+guideline|consensus\s+statement|expert\s+consensus|"
    r"editorial|commentary|letter\s+to\s+the\s+editor"
    r")\b"
)
RX_NIPT_SCREEN = re.compile(
    r"(?i)\b("
    r"non[- ]invasive\s+prenatal\s+(testing|screening|diagnosis)|"
    r"\bnipt\b|\bnips\b|cell[- ]free\s+(fetal\s+)?dna|"
    r"cfDNA\s+screening|cffDNA"
    r")\b"
)
RX_LARGE_N = re.compile(
    r"(?i)\b("
    r"\d{3,}\s+(singleton\s+|multiple\s+)?pregnanc(?:y|ies)|"
    r"\d{3,}\s+(cases|fetuses|patients|samples|amniocenteses)|"
    r"cohort\s+of\s+\d{3,}|"
    r"\d{4,}\s+pregnanc"
    r")\b"
)
RX_COHORT_SCREEN = re.compile(
    r"(?i)\b("
    r"cohort\s+study|retrospective\s+cohort|prospective\s+cohort|"
    r"screening\s+(accuracy|performance|for\s+aneuploid)|"
    r"population[- ]based\s+(study|cohort|screening)|"
    r"large[- ]scale\s+(cohort|screening|study)"
    r")\b"
)
RX_DB_PRED = re.compile(
    r"(?i)\b("
    r"genome[- ]wide\s+association|\bgwas\b|"
    r"polygenic\s+risk|pathogenicity\s+prediction|"
    r"in\s+silico\s+(prediction|analysis|tool)|"
    r"variant\s+database|clinvar\s+curation|"
    r"machine\s+learning\s+(model|classifier)|"
    r"deep\s+learning\s+(model|network)"
    r")\b"
)
RX_METHOD_TOOL = re.compile(
    r"(?i)\b("
    r"new\s+method|novel\s+assay|bioinformatic\s+pipeline|"
    r"software\s+(tool|package)|algorithm\s+for\s+detect"
    r")\b"
)

# --- protect true case reports ---------------------------------------------
RX_CASE_PROTECT = re.compile(
    r"(?i)\b("
    r"case\s+report|case\s+series|we\s+report|"
    r"proband|index\s+(patient|case)|"
    r"this\s+fetus|the\s+fetus\s+presented|"
    r"prenatal\s+diagnosis\s+of\s+a\s|"
    r"de\s+novo\s+(pathogenic\s+)?variant|"
    r"compound\s+heterozygous|segregation\s+analysis|"
    r"family\s+pedigree|trio\s+exome|"
    r"whole[- ]exome\s+sequencing\s+revealed|"
    r"identified\s+a\s+(novel|pathogenic)\s+variant"
    r")\b"
)

# design field (when available from labeled CSV)
DESIGN_HARD = re.compile(
    r"(?i)^(review|screening/?validation|method|cohort|retrospective\s+cohort|"
    r"prospective\s+cohort|observational|national\s+registry|diagnostic\s+validation)"
)


@dataclass
class RuleHit:
    code: str
    action: str  # force_irrelevant | demote_review
    detail: str = ""


@dataclass
class RuleResult:
    decision: str
    model_decision: str
    score: float
    threshold: float
    review_threshold: float
    rules_fired: list[RuleHit] = field(default_factory=list)
    protected_by_case: bool = False
    overridden: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rules_fired"] = [asdict(h) for h in self.rules_fired]
        d["rule_codes"] = [h.code for h in self.rules_fired]
        return d


def _model_decide(score: float, threshold: float, review_margin: float = REVIEW_MARGIN) -> str:
    if score >= threshold:
        return "relevant"
    if score >= threshold - review_margin:
        return "review"
    return "irrelevant"


def _head_text(title: str, abstract: str, text: str, max_chars: int = 12000) -> str:
    parts = [title or "", abstract or "", (text or "")[:max_chars]]
    return "\n".join(p for p in parts if p)


def collect_rule_hits(
    title: str,
    abstract: str,
    text: str = "",
    *,
    design: str = "",
) -> tuple[list[RuleHit], bool]:
    """Return (hits, protected_by_case)."""
    title = title or ""
    abstract = abstract or ""
    text = text or ""
    design = design or ""
    head = f"{title}\n{abstract}"
    blob = _head_text(title, abstract, text)
    protected = bool(RX_CASE_PROTECT.search(head) or RX_CASE_PROTECT.search(blob[:4000]))

    hits: list[RuleHit] = []

    # Design field: only hard types. Do NOT demote bare "cohort" — many true
    # positives are tagged cohort/case-series hybrids in this dataset.
    if design and re.search(r"(?i)^(review|method)\b", design.strip()):
        hits.append(RuleHit("DESIGN_HARD_EXCLUDE", "force_irrelevant", design.strip()[:60]))
    if design and re.search(r"(?i)screening/?validation", design.strip()) and not protected:
        hits.append(RuleHit("DESIGN_SCREENING", "force_irrelevant", design.strip()[:60]))

    # Title/abstract review / guideline
    if RX_REVIEW.search(title) or (RX_REVIEW.search(abstract) and not protected):
        hits.append(RuleHit("REVIEW_OR_GUIDELINE", "force_irrelevant", "title/abstract"))

    # NIPT / cfDNA screening at scale
    if RX_NIPT_SCREEN.search(head) and (RX_LARGE_N.search(head) or RX_COHORT_SCREEN.search(head)):
        hits.append(RuleHit("NIPT_SCREENING_COHORT", "force_irrelevant", "nipt+scale/cohort"))
    elif RX_NIPT_SCREEN.search(head) and not protected:
        hits.append(RuleHit("NIPT_SCREENING", "demote_review", "nipt"))

    # Large prenatal cohort / screening performance
    if RX_COHORT_SCREEN.search(head) and RX_LARGE_N.search(head) and not protected:
        hits.append(RuleHit("LARGE_PRENATAL_COHORT", "force_irrelevant", "cohort+n"))
    elif RX_COHORT_SCREEN.search(head) and not protected:
        hits.append(RuleHit("COHORT_LANGUAGE", "demote_review", "cohort"))

    # GWAS / database / in-silico prediction
    if RX_DB_PRED.search(head) and not protected:
        hits.append(RuleHit("GWAS_DB_OR_PREDICTION", "force_irrelevant", "prediction/gwas/db"))

    # Method/tool papers without case protect
    if RX_METHOD_TOOL.search(title) and not protected:
        hits.append(RuleHit("METHOD_TOOL_TITLE", "demote_review", "method title"))

    return hits, protected


def apply_rules(
    title: str,
    abstract: str,
    text: str,
    *,
    score: float,
    threshold: float,
    design: str = "",
    review_margin: float = REVIEW_MARGIN,
    enabled: bool = True,
) -> RuleResult:
    model_decision = _model_decide(score, threshold, review_margin)
    review_thr = threshold - review_margin
    if not enabled:
        return RuleResult(
            decision=model_decision,
            model_decision=model_decision,
            score=float(score),
            threshold=float(threshold),
            review_threshold=float(review_thr),
        )

    hits, protected = collect_rule_hits(title, abstract, text, design=design)
    decision = model_decision
    overridden = False

    # Apply force_irrelevant first (unless protected)
    force = [h for h in hits if h.action == "force_irrelevant"]
    demote = [h for h in hits if h.action == "demote_review"]

    if force and not protected:
        if decision != "irrelevant":
            overridden = True
        decision = "irrelevant"
    elif demote and not protected:
        if decision == "relevant":
            decision = "review"
            overridden = True
        # if already review/irrelevant, keep
    elif force and protected:
        # strong case cue: demote at most
        if decision == "relevant":
            decision = "review"
            overridden = True
            hits = hits + [RuleHit("PROTECTED_DEMOTE_ONLY", "demote_review", "case cue blocked hard exclude")]

    return RuleResult(
        decision=decision,
        model_decision=model_decision,
        score=float(score),
        threshold=float(threshold),
        review_threshold=float(review_thr),
        rules_fired=hits,
        protected_by_case=protected,
        overridden=overridden,
    )


def decide_with_rules(
    title: str,
    abstract: str,
    text: str,
    score: float,
    threshold: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: dict payload for predict/api."""
    return apply_rules(title, abstract, text, score=score, threshold=threshold, **kwargs).to_dict()
