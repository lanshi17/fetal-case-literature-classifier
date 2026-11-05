"""Load full text of a paper from its JATS XML, plain TXT or PDF and keep the most complete version."""
import re
from pathlib import Path

STUB_CHARS = 1500  # shorter than this = only a title/stub, no real full text


def xml_text(path):
    from lxml import etree
    root = etree.parse(str(path), etree.XMLParser(recover=True, huge_tree=True)).getroot()
    if root is None:
        return ""
    for bad in root.xpath("//ref-list|//back//ref|//table-wrap-foot|//fig//graphic"):
        bad.getparent().remove(bad)
    parts = []
    for xp in ("//article-title", "//abstract", "//body"):
        for el in root.xpath(xp):
            parts.append(" ".join(el.itertext()))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def txt_text(path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return re.sub(r"\s+", " ", Path(path).read_text(encoding=enc)).strip()
        except UnicodeDecodeError:
            continue
    return ""


def pdf_text(path, max_pages=40):
    import pymupdf
    try:
        with pymupdf.open(str(path)) as doc:
            return re.sub(r"\s+", " ", " ".join(p.get_text() for p, _ in zip(doc, range(max_pages)))).strip()
    except Exception:
        return ""


_READERS = {"xml": xml_text, "txt": txt_text, "pdf": pdf_text}


def best_text(files):
    """files: {'xml': path|None, 'txt': path|None, 'pdf': path|None} -> (text, source)."""
    best, src = "", "none"
    for kind in ("xml", "txt", "pdf"):
        p = files.get(kind)
        if p and Path(p).exists():
            try:
                t = _READERS[kind](p)
            except Exception:
                t = ""
            if len(t) > len(best):
                best, src = t, kind
    return best, src
