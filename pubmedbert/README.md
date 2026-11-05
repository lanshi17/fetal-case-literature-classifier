# casefilter

Screens a large literature set (10k+ papers) for UltraPhenoCards intake: keep a paper if it yields **at least one** fetal case that has a prenatal ultrasound phenotype abnormality and one of the 11 genetic tests (Karyotyping, CMA, FISH, CNV-seq, WES, Targeted Panel, MLPA, Sanger, WGS, PCR, mtDNA-seq).

## How it decides

Stage 1 (rules, offline, free, explainable) applies these gates to title + abstract (+ full text if provided):

| Gate | Meaning |
|---|---|
| hard exclude | review / meta-analysis / guideline / editorial publication type or title; animal or cell study with no human context; no PMID |
| FETAL | fetus / prenatal / pregnancy context |
| PHENO | ultrasound modality or a known sonographic finding |
| METHOD | one of the 11 tests (PCR and Sanger alone are weak evidence) |
| CASE | case report wording, individual fetus description, or a cohort with counts |
| LINK | one sentence mentions fetal context and a test together |

Labels: `INCLUDE` (all gates met), `REVIEW` (uncertain, e.g. abstract omits the test, or only PCR/Sanger), `EXCLUDE`. Recall is favoured: when unsure the paper goes to `REVIEW`, never to `EXCLUDE`.

Stage 2 (optional, `--llm`) sends only `REVIEW` papers to Claude with the intake criteria and caches answers in `llm.cache.jsonl` so runs resume. Low-confidence answers stay `REVIEW` for manual checking.

## Usage

```bash
pip install -e .            # add ".[llm]" for stage 2
casefilter run papers.csv --out out            # rules only
export ANTHROPIC_API_KEY=...                   # stage 2
casefilter run papers.csv --out out --llm
casefilter eval out/all.csv gold.csv           # recall / precision on a labelled subset
```

Input: `.csv/.tsv/.jsonl` with columns `pmid,title,abstract` (optional `pub_types`, `fulltext`), or a PubMed XML export. Output in `out/`: `all.csv`, `include.csv`, `review.csv`, `exclude.csv` with detected methods, reason codes and evidence sentence.

## Recommended workflow

1. Hand-label 200-300 papers (`pmid,label`), run `casefilter eval`, check false negatives first.
2. Tune patterns in `src/casefilter/lexicon.py` until recall is acceptable.
3. Run all papers with `--llm`; manually review what remains in `review.csv`.

Full text matters: abstracts often omit the tests used, which is why those papers land in `REVIEW`.

## Embedding classifier for full texts (scripts/)

Local PubMedBERT sentence embeddings (chunked full text, mean-pooled, plus first-chunk pool) feed a logistic-regression head that outputs 0/1.

```bash
python scripts/build_corpus.py --package <fulltext_package> --gold-dir <extracted> --gold-pdf-dir <gold pdfs> --out data/corpus.jsonl
python scripts/train_and_classify.py --model <local NeuML/pubmedbert-base-embeddings dir> --out out
python scripts/summarize.py --out out      # summary.md + summary.png
```

Training labels: curated papers = 1; papers the rule stage confidently excludes = 0. The head is trained with source-balanced weights so the text source (XML/TXT/PDF) cannot act as a shortcut. Training papers get out-of-fold predictions. Data and outputs are git-ignored.

## Standalone inference (no Claude)

`weights/` (created by `python scripts/export_weights.py`) holds the exported head: `head_v3.npz` (1536 coefficients + intercept), `config.json` (embedding model, chunking, pooling, threshold, validation metrics). Classify any folder of xml/txt/pdf full texts:

```bash
casefilter predict <folder> --weights weights --model <local NeuML/pubmedbert-base-embeddings dir> --out out/predict.csv
```

Output columns: file, pmid, source, n_chars, stub, prob, label (prob >= 0.5). Re-running on 12 training-set papers reproduced the stored probabilities within 0.044 and all 12 labels.
