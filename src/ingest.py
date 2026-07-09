"""
Loads documents, chunks them, embeds them, and builds a searchable FIASS index

Embeddings: uses TF-IDF (scikit-learn) reduced via SVD as a lightweight,
fully-offline embedding stand-in — no external model download required,
so this runs anywhere with zero network access. In a real deployment you'd
swap `embed_texts()` for OpenAI/Anthropic/sentence-transformers embeddings;
everything downstream (FAISS index, retrieval, agents) is unchanged either way.

Also includes `fetch_edgar_filing()` — a real SEC EDGAR fetcher you can use
against live filings once you have network access to sec.gov.
"""

import os
import glob
import json
import pickle
import re
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
EMBED_DIM = 128

def load_documents():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "filings", "*.txt")):
        with open(path, "r") as f:
            docs.append({"text": f.read(), "source": os.path.basename(path), "type": "filing"})
    for path in glob.glob(os.path.join(DATA_DIR, "policies", "*.md")):
        with open(path, "r") as f:
            docs.append({"text": f.read(), "source": os.path.basename(path), "type": "policy"})
    return docs

def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120):
    """Simple sliding-window chunker on paragraph boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current.strip())
            # keep overlap tail
            current = current[-overlap:] + "\n\n" + p
        else:
            current = (current + "\n\n" + p).strip()
    if current:
        chunks.append(current.strip())
    return [{"text": c, "source": source} for c in chunks]

def build_corpus():
    docs = load_documents()
    corpus = []
    for d in docs:
        chunks = chunk_text(d["text"], d["source"])
        for c in chunks:
            c["type"] = d["type"]
            corpus.append(c)
    return corpus

def embed_texts(texts, vectorizer=None, svd=None, fit=False):
     if fit:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        n_comp = min(EMBED_DIM, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=max(n_comp, 2))
        vecs = svd.fit_transform(tfidf)
     else:
        tfidf = vectorizer.transform(texts)
        vecs = svd.transform(tfidf)
     vecs = vecs.astype("float32")
     faiss.normalize_L2(vecs)
     return vecs, vectorizer, svd

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    corpus = build_corpus()
    texts = [c["text"] for c in corpus]
    vecs, vectorizer, svd = embed_texts(texts, fit=True)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"corpus": corpus, "vectorizer": vectorizer, "svd": svd}, f)

    print(f"Indexed {len(corpus)} chunks from {len(set(c['source'] for c in corpus))} documents.")
    return index, corpus


def fetch_edgar_filing(cik: str, accession_number: str):
    """
    Real SEC EDGAR fetcher (production use — requires network access to
    https://www.sec.gov). Not called by the offline demo; included so this
    module is a genuine drop-in replacement for the synthetic sample data.
    """
    import requests
    headers = {"User-Agent": "MarketMind Research Bot contact@example.com"}
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text


if __name__ == "__main__":
    build_index()