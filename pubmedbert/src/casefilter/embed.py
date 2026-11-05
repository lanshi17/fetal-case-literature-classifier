"""Document embedding: split full text into token chunks and embed each chunk locally."""
import numpy as np

DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"  # PubMedBERT sentence embeddings, 512-token window


def chunk_text(text, tokenizer, size=480, max_chunks=24):
    """Token-window chunks. Always keeps the first 2 (title/abstract), samples the rest evenly."""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]
    if len(chunks) > max_chunks:
        rest = np.linspace(2, len(chunks) - 1, max_chunks - 2).round().astype(int)
        chunks = chunks[:2] + [chunks[i] for i in rest]
    return [tokenizer.decode(c) for c in chunks]


class Embedder:
    def __init__(self, model=DEFAULT_MODEL, device=None, batch_size=32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = 512
        self.bs = batch_size

    def embed_docs(self, texts, max_chunks=24):
        tok = self.model.tokenizer
        per_doc = [chunk_text(t or " ", tok, max_chunks=max_chunks) for t in texts]
        flat = [c for d in per_doc for c in d]
        vecs = self.model.encode(flat, batch_size=self.bs, normalize_embeddings=True, show_progress_bar=True)
        out, k = [], 0
        for d in per_doc:
            out.append(vecs[k:k + len(d)])
            k += len(d)
        return out  # one (n_chunks, dim) array per document


def pool(chunk_vecs, head=2):
    """Document feature = [mean over all chunks, mean over the first `head` chunks (title/abstract)]."""
    f = []
    for v in chunk_vecs:
        a, b = v.mean(axis=0), v[:head].mean(axis=0)
        f.append(np.concatenate([a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)]))
    return np.vstack(f)
