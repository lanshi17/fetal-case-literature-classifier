"""Term lexicons. All patterns are case-insensitive regexes applied to title+abstract(+fulltext)."""
import re

I = re.IGNORECASE


def _c(p):
    return re.compile(p, I)


# The 11 detection schemes of the database (D03-D13).
METHODS = {
    "Karyotyping": _c(r"karyotyp\w*|G-band\w*|chromosom(?:e|al) (?:analysis|culture|abnormalit\w+)|cytogenetic\w* (?:analysis|testing|study)"),
    "CMA": _c(r"chromosomal microarray|\bmicroarray\b|\bCMA\b|array[- ]?CGH|\baCGH\b|SNP[- ]array|comparative genomic hybridi[sz]ation"),
    "FISH": _c(r"\bFISH\b|fluorescen\w* in situ hybridi[sz]ation"),
    "CNV-seq": _c(r"CNV[- ]?seq\w*|copy number variation sequencing|low[- ](?:pass|coverage) (?:whole[- ]genome )?sequencing"),
    "WES": _c(r"\bWES\b|exome sequencing|\bexome\b|trio[- ]ES\b|clinical exome"),
    "Targeted Panel": _c(r"(?:gene|targeted|multigene|NGS|disease) panel|targeted (?:gene |next[- ]generation )?sequencing"),
    "MLPA": _c(r"\bMLPA\b|multiplex ligation"),
    "Sanger": _c(r"\bSanger\b"),
    "WGS": _c(r"\bWGS\b|(?<!low-pass )(?<!low pass )(?<!low-coverage )whole[- ]genome sequencing|\bgenome sequencing\b"),
    "PCR": _c(r"\bPCR\b|polymerase chain reaction|QF-PCR|\bqPCR\b|PCR-RFLP"),
    "mtDNA-seq": _c(r"mitochondrial DNA|\bmtDNA\b|mitochondrial genome"),
}
# Methods that appear in nearly every molecular paper's methods section; weak evidence on their own.
WEAK_METHODS = {"PCR", "Sanger"}

FETAL = _c(r"\bfet(?:us|uses|al|i)\b|prenatal\w*|antenatal\w*|in utero|intrauterine|amniocentesis|chorionic villus|cordocentesis|"
           r"termination of pregnancy|pregnan\w+|gravida|gestation\w*|embryo\w*")
US_MODALITY = _c(r"ultrasound|ultrasonograph\w*|sonograph\w*|echocardiograph\w*|\bsonogram\b|\bUS\b scan|prenatal imaging|fetal MRI")
US_FINDING = _c(
    r"nuchal translucency|\bNT\b (?:thickness|increas\w+|>)|increased NT|nuchal (?:fold|edema|thickening)|cystic hygroma|hydrops|"
    r"ventriculomegaly|hydrocephalus|dandy[- ]walker|agenesis of the corpus callosum|corpus callosum|holoprosencephaly|"
    r"congenital heart (?:defect|disease)|cardiac (?:defect|anomal\w+|malformation)|tetralogy|hypoplastic left heart|"
    r"ventricular septal defect|atrioventricular septal|coarctation|"
    r"absent nasal bone|nasal bone|echogenic (?:bowel|intracardiac focus|focus)|pyelectasis|hydronephrosis|"
    r"skeletal dysplasia|short (?:femur|long bones?)|femur length|polydactyly|clubfoot|talipes|"
    r"cleft (?:lip|palate)|diaphragmatic hernia|omphalocele|gastroschisis|"
    r"polyhydramnios|oligohydramnios|fetal growth restriction|\bFGR\b|\bIUGR\b|single umbilical artery|"
    r"multicystic dysplastic kidney|renal agenesis|megacystis|choroid plexus cyst|"
    r"structural (?:anomal\w+|malformation\w*|abnormalit\w+)|congenital (?:anomal\w+|malformation\w*)|"
    r"(?:ultrasound|sonographic) (?:anomal\w+|abnormalit\w+|finding\w*|marker\w*|phenotype\w*)|soft markers?|"
    r"fetal (?:anomal\w+|malformation\w*|abnormalit\w+)"
)
# Evidence that individual fetuses/pregnancies are described.
CASE_STRONG = _c(
    r"case report|case series|we (?:report|present|describe)|here(?:in)? we (?:report|present|describe)|"
    r"a (?:\d+|\w+)[- ]year[- ]old (?:pregnant |gravida|woman|primigravida)|\bgravida \d|\bG\d ?P\d|primigravid\w*|"
    r"(?:the|this|a|one|our) (?:\w+ )?fet(?:us|al)\b|\bproband\b|pedigree|family \d|case \d|patient \d|"
    r"termination of pregnancy|pregnancy was terminated|autops\w+|born at|delivered at|live[- ]?born|stillbirth"
)
CASE_COHORT = _c(r"\b\d+ (?:fetuses|fetal cases|pregnancies|cases|fetus|families|couples|singleton|foetuses|women)\b|"
                 r"(?:retrospective|cohort|consecutive)")
# Hard negatives
REVIEW_TITLE = _c(r"^\W*(?:a )?(?:systematic|literature|narrative|comprehensive)? ?(?:review|meta-analysis|guideline\w*|consensus|practice (?:bulletin|guideline))|"
                  r"\b(?:systematic review|meta-analysis|review of the literature|literature review|a review)\b")
REVIEW_PUBTYPE = _c(r"review|meta-analysis|guideline|practice guideline|editorial|comment|letter|news|retraction")
CASE_PUBTYPE = _c(r"case reports?")
ANIMAL = _c(r"\bmice\b|\bmouse\b|murine|zebrafish|\brats?\b|knock[- ]?out model|drosophila|xenopus|porcine|bovine|\bsheep\b|\bcell lines?\b|in vitro")
HUMAN = _c(r"\bhuman\b|pregnant wom[ae]n|\bgravida|\bpatients?\b|\bfetuses\b|\bfetus\b|couples?")
