"""Stage 1: deterministic, explainable rule classifier.

Intake rule (pubmed-search-strategy 1.1, D15 7.2): a paper qualifies if it describes at least one
human fetus with (a) an abnormal prenatal ultrasound phenotype, (b) at least one of the 11 genetic
tests linked to that fetus, (c) is not a review / animal study. Recall is prioritised: uncertain
papers get label REVIEW instead of EXCLUDE.
"""
from dataclasses import dataclass, field, asdict
import re

from . import lexicon as L

INCLUDE, REVIEW, EXCLUDE = "INCLUDE", "REVIEW", "EXCLUDE"

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Decision:
    pmid: str
    label: str
    score: float
    methods: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    case_type: str = ""  # "case" | "cohort" | ""
    stage: str = "rules"

    def to_dict(self):
        d = asdict(self)
        d["methods"] = ";".join(self.methods)
        d["reasons"] = ";".join(self.reasons)
        d["evidence"] = " || ".join(self.evidence)
        return d


def _methods_in(text):
    return [m for m, rx in L.METHODS.items() if rx.search(text)]


def classify_rules(rec: dict) -> Decision:
    """rec keys: pmid, title, abstract, pub_types (str, optional), fulltext (optional)."""
    pmid = str(rec.get("pmid", "")).strip()
    title = rec.get("title") or ""
    abstract = rec.get("abstract") or ""
    fulltext = rec.get("fulltext") or ""
    pubtypes = rec.get("pub_types") or ""
    has_ft = len(fulltext) > 2000
    text = "\n".join(x for x in (title, abstract, fulltext) if x)
    reasons, evidence = [], []

    def out(label, score, methods=(), case_type=""):
        return Decision(pmid, label, round(score, 2), list(methods), reasons, evidence[:4], case_type)

    if not pmid:
        reasons.append("NO_PMID")
        return out(EXCLUDE, 0)
    if not (title or abstract):
        reasons.append("NO_TEXT")
        return out(REVIEW, 0)

    # --- hard exclusions -------------------------------------------------
    is_case_pt = bool(L.CASE_PUBTYPE.search(pubtypes))
    if L.REVIEW_PUBTYPE.search(pubtypes) and not is_case_pt:
        reasons.append("REVIEW_PUBTYPE")
        return out(EXCLUDE, 0)
    if L.REVIEW_TITLE.search(title) and not is_case_pt:
        reasons.append("REVIEW_TITLE")
        return out(EXCLUDE, 0)
    if L.ANIMAL.search(text) and not L.HUMAN.search(text):
        reasons.append("ANIMAL_ONLY")
        return out(EXCLUDE, 0)

    # --- gates ------------------------------------------------------------
    g_fetal = bool(L.FETAL.search(text))
    g_modality = bool(L.US_MODALITY.search(text))
    findings = L.US_FINDING.search(text)
    g_pheno = g_modality or bool(findings)
    methods = _methods_in(text)
    strong_methods = [m for m in methods if m not in L.WEAK_METHODS]
    g_method = bool(methods)
    case_strong = L.CASE_STRONG.search(text) or is_case_pt
    case_cohort = L.CASE_COHORT.search(text)
    g_case = bool(case_strong or case_cohort)

    # A method is "linked" when a sentence mentions fetal context and the method together.
    linked = False
    for s in _SENT.split(text):
        if L.FETAL.search(s) and any(L.METHODS[m].search(s) for m in methods):
            linked = True
            evidence.append(s.strip()[:240])
            break
    if findings:
        evidence.insert(0, findings.group(0))

    score = (1.0 * g_fetal + 1.0 * g_pheno + 0.5 * g_modality + 1.0 * bool(strong_methods)
             + 0.5 * g_method + 1.0 * bool(case_strong) + 0.5 * bool(case_cohort) + 1.0 * linked)
    case_type = "case" if case_strong else ("cohort" if case_cohort else "")
    for name, ok in (("FETAL", g_fetal), ("PHENO", g_pheno), ("METHOD", g_method), ("CASE", g_case), ("LINK", linked)):
        reasons.append(("HAS_" if ok else "NO_") + name)
    if methods and not strong_methods:
        reasons.append("WEAK_METHOD_ONLY")

    if not g_fetal or not g_pheno:
        return out(EXCLUDE, score, methods)

    # Abstract-only input often omits methods or case detail: send to REVIEW rather than lose a paper.
    if g_method and g_case and linked:
        if strong_methods:
            return out(INCLUDE, score, methods, case_type)
        return out(REVIEW, score, methods, case_type)  # PCR/Sanger only
    if g_method and g_case:
        return out(REVIEW, score, methods, case_type)
    if not has_ft and g_case and (g_modality or findings) and not g_method:
        reasons.append("METHOD_MAY_BE_IN_FULLTEXT")
        return out(REVIEW, score, methods, case_type)
    if g_method and linked and g_pheno and not g_case:
        reasons.append("CASE_DETAIL_UNCLEAR")
        return out(REVIEW, score, methods, case_type)
    return out(EXCLUDE, score, methods, case_type)
