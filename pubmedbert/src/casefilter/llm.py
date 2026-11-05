"""Stage 2 (optional): LLM adjudication for papers the rules label REVIEW.

Needs `pip install anthropic` and ANTHROPIC_API_KEY. Results are cached in a JSONL file so a
10k-paper run can be resumed without paying twice.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .rules import INCLUDE, EXCLUDE, REVIEW

DEFAULT_MODEL = "claude-haiku-5-5"

CRITERIA = """You screen papers for a prenatal database. A paper QUALIFIES if it reports at least ONE individual human fetus (or pregnancy) such that ALL hold:
1. A prenatal ultrasound phenotype abnormality is described (structural anomaly or soft marker, e.g. ventriculomegaly, increased NT, cardiac defect).
2. At least one of these genetic tests was performed and is clearly linked to that fetus (positive or negative result): Karyotyping, CMA, FISH, CNV-seq, WES, Targeted Panel, MLPA, Sanger, WGS, PCR, mtDNA-seq.
3. The paper is primary research or a case report (NOT a narrative/systematic review, meta-analysis, guideline, animal/cell study, or pure method paper).
Cohort studies qualify only if they describe the ultrasound findings and test results of the studied fetuses (not just aggregate screening accuracy).
When the abstract is genuinely ambiguous, answer qualifies=true with low confidence (recall matters more than precision)."""

SCHEMA_HINT = ('Reply with ONLY JSON: {"qualifies": bool, "confidence": 0-1, "methods": [..from the 11 names..], '
               '"case_type": "case"|"cohort"|"none", "reason": "<=30 words"}')


def _prompt(rec, max_chars=12000):
    body = (rec.get("fulltext") or rec.get("abstract") or "")[:max_chars]
    return (f"{CRITERIA}\n\n{SCHEMA_HINT}\n\nTITLE: {rec.get('title', '')}\n"
            f"PUBLICATION TYPES: {rec.get('pub_types', '')}\nTEXT:\n{body}")


def parse_reply(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


class LLMJudge:
    def __init__(self, model=DEFAULT_MODEL, cache="llm.cache.jsonl", workers=8, client=None):
        if client is None:
            import anthropic  # lazy: stage 1 works without the SDK
            client = anthropic.Anthropic()
        self.client, self.model, self.workers = client, model, workers
        self.cache_path = Path(cache)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["pmid"]] = d["result"]

    def _ask(self, rec):
        msg = self.client.messages.create(
            model=self.model, max_tokens=400,
            messages=[{"role": "user", "content": _prompt(rec)}])
        return parse_reply("".join(b.text for b in msg.content if getattr(b, "text", None)))

    def judge(self, rec):
        pmid = str(rec["pmid"])
        if pmid not in self.cache or (self.cache[pmid] or {}).get("error"):
            try:
                res = self._ask(rec)
            except Exception as e:  # network / parse failure: keep for manual review
                res = {"error": str(e)[:200]}
            self.cache[pmid] = res
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pmid": pmid, "result": res}, ensure_ascii=False) + "\n")
        return self.cache[pmid]

    def adjudicate(self, pairs, min_conf=0.5):
        """pairs: [(rec, Decision)] -> list of updated Decisions."""
        def one(p):
            rec, dec = p
            r = self.judge(rec)
            if not r or "qualifies" not in r:
                dec.reasons.append("LLM_FAILED")
                return dec
            conf = float(r.get("confidence", 0))
            dec.stage = "llm"
            dec.reasons.append("LLM:" + str(r.get("reason", ""))[:120])
            dec.methods = r.get("methods") or dec.methods
            if r.get("case_type") in ("case", "cohort"):
                dec.case_type = r["case_type"]
            if r["qualifies"]:
                dec.label = INCLUDE if conf >= min_conf else REVIEW
            else:
                dec.label = EXCLUDE if conf >= min_conf else REVIEW
            return dec
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, pairs))
