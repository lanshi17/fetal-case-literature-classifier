# Boundary annotation guide (case report / case series / cohort)

Purpose: adjudicate the 209-row sheet `artifacts/label_audit/boundary_set.csv` (all 97 current
case series, a 100-row cohort sample, and 26 structural flags). Human decisions only; do not
edit `data/labeled_papers.csv` directly.

## Definitions

| label | definition |
|---|---|
| `case_report` | one patient / fetus / pregnancy, or one family reported as a single narrative |
| `case_series` | ≥2 cases collected and described together, **no comparison group**, no incidence estimate |
| `cohort` | a defined group with a comparison, follow-up, or incidence/association estimate (retrospective or prospective) |
| `screening` | population screening performance (NIPT, cfDNA, ultrasound), even when detected cases are described individually |
| `validation` | assay/classifier validation study |
| `registry` | registry-based collection as the primary design |
| `review` / `method` | literature review / technique paper |
| `mixed` | two designs equally central; record the primary one in notes |
| `unclear` | insufficient information to decide |

## Decision rules (apply in order)

1. **Population-level denominator?** If the paper reports screening/diagnostic yield over a
   defined population (e.g. 19,607 pregnancies), label `screening` or `cohort` — not case series.
2. **Comparison or controls?** Any comparator group (affected vs unaffected, ES vs CMA,
   case vs control) → `cohort`/`validation`.
3. **How many cases, how reported?** Single narrative → `case_report`; ≥2 with individual or
   aggregate description and no denominator → `case_series`.
4. **Author wording is evidence, not ground truth.** "Case series" in the title does not override
   a population denominator or a comparison group.
5. **Size alone does not decide.** A 200-case descriptive collection without controls remains a
   case series; a 20-case study with controls is a cohort.
6. **Familial studies**: multiple affected relatives described individually → `case_series`
   (note "familial"); one family narrative → `case_report`.
7. **Ambiguous** → `unclear`, confidence `low`, and quote the conflicting sentences.

## Fields to fill

| column | values |
|---|---|
| `annotator_design` | one of the labels above |
| `annotator_confidence` | `high` / `medium` / `low` |
| `annotator_notes` | one sentence + a short verbatim quote |

Use `title`, `abstract` (in `data/labeled_papers.csv`), and Methods when the abstract is
insufficient. Two annotators on a 30-row overlap; disagreements resolved by a third.

## After annotation

1. Merge agreed rows into an overlay CSV (`pmid, design, source=audit_v1`).
2. Re-run `design_taxonomy.py --overlay` and `finetune_case_dora.py` with the overlay.
3. Report the delta in CV metrics and the changed case counts; do not mix audited and unaudited
   labels without a `label_source` column.
