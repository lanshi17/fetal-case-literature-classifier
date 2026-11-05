import json
from pathlib import Path

from casefilter.rules import classify_rules, INCLUDE, REVIEW, EXCLUDE
from casefilter.llm import LLMJudge

SAMPLE = Path(__file__).parent.parent / "examples" / "sample.jsonl"


def _load():
    return {r["pmid"]: r for r in map(json.loads, SAMPLE.read_text(encoding="utf-8").splitlines())}


def test_sample_labels():
    recs = _load()
    expect = {"1001": INCLUDE, "1002": INCLUDE, "1003": EXCLUDE, "1004": EXCLUDE,
              "1005": EXCLUDE, "1006": REVIEW, "1007": EXCLUDE}
    got = {p: classify_rules(r).label for p, r in recs.items()}
    assert got == expect


def test_methods_detected():
    d = classify_rules(_load()["1001"])
    assert {"Karyotyping", "CMA"} <= set(d.methods)
    assert d.case_type == "case"


def test_missing_pmid_excluded():
    assert classify_rules({"pmid": "", "title": "x", "abstract": "y"}).label == EXCLUDE


def test_llm_resolves_review(tmp_path):
    class Msg:
        content = [type("B", (), {"text": '{"qualifies": true, "confidence": 0.9, "methods": ["WES"], '
                                          '"case_type": "case", "reason": "ok"}'})()]

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                return Msg()

    rec = _load()["1006"]
    dec = classify_rules(rec)
    out = LLMJudge(cache=str(tmp_path / "c.jsonl"), client=Client()).adjudicate([(rec, dec)])
    assert out[0].label == INCLUDE and out[0].stage == "llm"
