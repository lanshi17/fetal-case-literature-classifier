# Extraction schema (P3, spec only)

The current release stops at study-design classification. This document specifies the next
stage — schema-guided extraction with evidence grounding — and is **not implemented**. Nothing
in the model card or API produces these fields today.

## Target record

```json
{
  "pmid": "12345678",
  "study_design": {"label": "case_report", "evidence": {"quote": "We report a fetus with ...", "section": "Abstract"}},
  "prenatal": {
    "gestational_age_weeks": {"value": 23, "evidence": {"quote": "at 23 weeks and 6 days", "section": "Case presentation"}},
    "phenotype": [{"hpo_term": "Hydrops fetalis", "hpo_id": "HP:0001789", "evidence": {"quote": "nonimmune hydrops fetalis", "section": "Results"}}]
  },
  "variant": {
    "gene": "PIEZO1",
    "hgvs": ["NM_001142864.4:c.7367G>A", "p.Arg2456His"],
    "zygosity": "compound heterozygous",
    "inheritance": "autosomal recessive",
    "evidence": {"quote": "...", "section": "Molecular analysis"}
  },
  "testing": [{"method": "whole-exome sequencing", "evidence": {"quote": "...", "section": "Methods"}}],
  "outcome": {"pregnancy": "terminated at 24 weeks", "evidence": {"quote": "...", "section": "Discussion"}},
  "confidence": 0.82,
  "review_status": "unreviewed"
}
```

## Grounding rules

1. Every extracted value carries a verbatim `quote` and a section name; offsets are recomputed
   against the source text and rejected if not found.
2. Values absent from the text are `null`, never imputed.
3. Contradictory statements across sections are kept as separate evidence items and flagged.
4. Study design is copied from the existing classifier only when the classifier is `case`; the
   extraction layer may override it with evidence.

## Evaluation plan

- Field-level precision/recall/F1 against a double-annotated sample (≥100 case reports).
- Span exactness (character-level overlap) and hallucination rate (quotes not found in the source).
- HPO/HGVS normalization accuracy separately from extraction recall.
- Cost/stability across models (specialized encoder vs LLM vs ensemble) on the same sample.

## Status

| item | status |
|---|---|
| schema | defined here |
| annotator guide | not written |
| extractor | not trained |
| evaluation set | not annotated |

Do not treat this file as evidence that extraction works.
