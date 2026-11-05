# DoRA review export

- threshold=0.8794 (val R≥0.95)
- gold: n=40 FN=3 FP=16
- test: n=177 FN=4 FP=30
- review_queue rows=28 (gold errors first; test errors excluding existing gold)

## Gold FN
- 31638247 score=0.779 | Novel compound heterozygous PKHD1 mutations cause autosomal recessive polycystic
- 42682406 score=0.873 | Case Report: unmasking pseudo-homozygosity in &lt;i&gt;CYP21A2&lt;/i&gt;: interg
- 37422672 score=0.768 | Clear cell endometrial carcinoma with high microsatellite instability in a compl

## Gold FP
- 32149439 score=0.931 | Neurodevelopmental outcome at 2 years of corrected age in fetuses with increased
- 40392900 score=0.889 | Karyotyping with amniotic fluid in 6,572 pregnant women and pregnancy outcomes--
- 40992929 score=0.924 | Residual Risks of Fetal Chromosome Aberrations When Cell-Free DNA Prenatal Scree
- 33785045 score=0.917 | Performance of cell-free DNA sequencing-based non-invasive prenatal testing: exp
- 35195919 score=0.930 | Combined diagnosis of QF-PCR and CNV-Seq in fetal chromosomal abnormalities: A n
- 17171632 score=0.931 | First-trimester combined screening for trisomy 21 in a predominantly Chinese pop
- 38284447 score=0.948 | Clinical value of positive CNVs results by NIPT without fetal ultrasonography-id
- 28154732 score=0.949 | [Meckel Gruber syndrome: about a rare case].
- 38687434 score=0.931 | ZAP70: A Key Gene Identified by Differential Expression Analysis for Early Diagn
- 39198908 score=0.887 | MRI in the prenatal genetic diagnosis and intrauterine treatment of fetal congen
- 35902388 score=0.941 | Application of the prenatal BACs-on-Beads™ assay for rapid prenatal detection of
- 36636555 score=0.949 | Assessment of Combined Karyotype Analysis and Chromosome Microarray Analysis in 
- 32864771 score=0.942 | The difference between karyotype analysis and chromosome microarray for mosaicis
- 14726642 score=0.953 | Prenatal sonographic diagnosis of focal musculoskeletal anomalies.
- 25923652 score=0.951 | Fetal calcifications are associated with chromosomal abnormalities.
- 31523615 score=0.953 | A Systematic Clinical Review of Prenatally Diagnosed Tetrasomy 9p.

## How to review

```bash
.venv/bin/python scripts/gold_review.py \
  --predictions /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/artifacts/finetune_fair/dora/predictions_for_review.csv \
  --fn /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/artifacts/finetune_fair/dora/fn_audit.csv --fp /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/artifacts/finetune_fair/dora/fp_audit.csv \
  --queue fn_fp --port 8765
```

Or open `review_queue.csv` / `REVIEW.md` offline.
