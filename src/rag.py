"""
Retrieval + grounded answer generation. Every answer must cite the specific
source chunk(s) it used; if retrieval confidence is too low, the system says
so instead of guessing (a basic hallucination guard).
"""

import os
import sys
import pickle
import faiss
import numpy as np

from src.llm import call_llm

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

_SYSTEM_PROMPT = """You are a financial research assistant used internally by
quantitative analysts and compliance officers. Answer ONLY using the provided
context chunks. Every factual claim must reference the source filename in
brackets, e.g. [ACME_10K_2025.txt]. If the context does not contain enough
information to answer confidently, say so explicitly rather than guessing.
Never fabricate numbers, dates, or filing details."""

class RetrievalIndex:
    def __init__(self):
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.corpus = meta["corpus"]
        self.vectorizer = meta["vectorizer"]
        self.svd = meta["svd"]

    def search(self, query: str, k: int = 4, min_score: float = 0.05):
        from src.ingest import embed_texts
        vec, _, _ = embed_texts([query], vectorizer=self.vectorizer, svd=self.svd, fit=False)
        scores, idxs = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or score < min_score:
                continue
            chunk = self.corpus[idx]
            results.append({**chunk, "score": float(score)})
        return results


def answer_question(query: str, index: "RetrievalIndex", k: int = 4) -> dict:
    hits = index.search(query, k=k)
    if not hits:
        return {
            "answer": "I don't have enough grounded information in the indexed "
                    "documents to answer that confidently.",
            "sources": [],
            "grounded": False,
        }

    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer with citations."
    answer = call_llm(_SYSTEM_PROMPT, user_prompt, role="research")
    return {
        "answer": answer,
        "sources": [{"source": h["source"], "score": round(h["score"], 3)} for h in hits],
        "grounded": True,
    }

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    idx = RetrievalIndex()
    for q in [
        "What are ACME Robotics' main risk factors?",
        "What is Beacon Biotech's cash runway and going concern situation?",
        "What is Harbor Financial's CET1 capital ratio?",
    ]:
        print("Q:", q)
        result = answer_question(q, idx)
        print("A:", result["answer"])
        print("Sources:", result["sources"])
        print("-" * 60)