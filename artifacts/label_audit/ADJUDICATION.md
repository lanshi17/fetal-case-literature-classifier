# Label adjudication (AI first pass — human sign-off required)

Source: `review_queue.csv` (26 structural flags). Suggestions below are derived from titles and
abstracts only; **`data/labeled_papers.csv` is unchanged**. Apply only after expert review.

Actions: `add` = currently excluded (design missing/other) but declares a case study;
`relabel` = design conflicts with the abstract; `confirm` = flag is a false alarm.

| PMID | current | suggested | action | evidence |
|---|---|---|---|---|
| 18955803 | (missing) | case_report | add | “Prenatal diagnosis of fetal seizure: a case report”; single fetus |
| 35791503 | (missing) | case_series | add | “three case reports” / four fetuses from three families |
| 36609239 | (missing) | case_report | add | “a case report”; monozygotic twins |
| 37761271 | (missing) | case_report | add | “Skeletal Dysplasia: A Case Report” |
| 40761308 | (missing) | case_report | add | “a case report”; one pregnancy |
| 42017141 | (missing) | case_report | add | “Case Report:” triple-line mosaicism |
| 42116066 | other | case_report | add | “case report”; one newborn |
| 41029435 | (missing) | case_series | add | “a retrospective case series”; 20 cases |
| 41710465 | case series | case_series | confirm | title says comparative case series; 250 is a referral-center count |
| 33545980 | case series | case_series | confirm | eight pregnancies; 150 is a genomic coordinate |
| 34587985 | case series | case_series | confirm | cytogenetic case analysis; no cohort design stated |
| 38259626 | case series | case_series | confirm | 12 sSMC cases; 620 is a coordinate |
| 42795903 | case series | case_series | confirm | “A 15-Case Series” |
| 41761011 | case series | case_series | confirm | multi-center case series (37) |
| 41716663 | case report | case_report | confirm | single fetus, “Case Report” |
| 41494975 | cohort | case_series | relabel | title: “A Retrospective Case Series” (20 CNVs) |
| 26260800 | case series | screening | relabel | cfDNA false-negative series; screening study |
| 32158503 | case series | screening | relabel | NIPS validation for Emanuel syndrome |
| 32067421 | case series | screening | relabel | 975 low-risk NIPS false negatives |
| 39262391 | case series | screening | relabel | NIPS in 19,607 pregnancies |
| 36553582 | case series | cohort | relabel | 4,968 invasive diagnoses; 16p11.2 subset |
| 30187503 | case series | cohort | relabel | placental UPD cohort (TRIDENT) |
| 41751555 | case series | cohort | relabel | retrospective SCA series (67) |
| 40041231 | case series | cohort | relabel | CAKUT counseling cohort |
| 35229910 | cohort | cohort | confirm | retrospective ES vs CMA study (114) |
| 41267400 | cohort | cohort | confirm | 178 fetuses among 20,546 CMAs |

## Impact if the 9 in-corpus relabels were applied

| metric | reported | relabelled |
|---|---:|---:|
| OOF AP | 0.9650 | 0.9661 |
| OOF AUC | 0.9699 | 0.9713 |
| F1 @ 0.396 | 0.9443 | 0.9398 |

Label noise in the *case series* band **does not inflate** the reported ranking metrics; it shifts
F1 by −0.005. The larger issue is coverage: 8 case reports were excluded from training because
`design` was missing/`other`.

## Impact if the 8 excluded case reports/series are added

Scores from the released model (CPU): 7 of 8 score `case` (0.922–0.934); the 20-case
retrospective series `41029435` scores **0.322** (`non-case`). Full overlay (872 rows, 377
positives): AP **0.9671**, AUC **0.9714**, P 0.930 / R 0.950 / F1 **0.940**, versus
0.9650 / 0.9699 / 0.9443 on the original 864. Adding the excluded positives barely moves the
ranking and exposes one real failure mode: small retrospective case series with cohort-like
wording.

## Next step

An expert reviews these 26 rows (≈1–2 h). Confirmed rows go into a `design_audit` overlay, then the
model is retrained and the CV repeated. Until then, treat the binary labels as noisy.
