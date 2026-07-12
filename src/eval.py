"""
Lightweight, dependency-free evaluation harness inspired by RAGAS metrics.
Computes, per question:
  - context_precision@k : did retrieval surface the expected source doc?
  - keyword_recall       : are the expected factual keywords present in the
                           retrieved context (proxy for "could the answer be
                           grounded")?
  - answer_has_citation  : did the generated answer include a bracketed
                           citation at all?

This runs fully offline (no LLM/RAGAS API calls needed) so it can run in
CI on every PR. Swap in real `ragas` + an LLM judge for production-grade
faithfulness/answer-relevancy scoring once you have API access — the
question set and pass/fail gate structure stay the same.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag import RetrievalIndex, answer_question


QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "eval_questions.json")

THRESHOLDS = {"context_precision": 0.80, "keyword_recall": 0.75}

def run_eval():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    index = RetrievalIndex()
    results = []
    for q in questions:
        hits = index.search(q["question"], k=4)
        retrieved_sources = {h["source"] for h in hits}
        retrieved_text = " ".join(h["text"].lower() for h in hits)

        context_hit = q["expected_source"] in retrieved_sources
        keyword_hits = [kw for kw in q["expected_keywords"] if kw.lower() in retrieved_text]
        keyword_recall = len(keyword_hits) / len(q["expected_keywords"])

        rag_result = answer_question(q["question"], index)
        has_citation = "[" in rag_result["answer"] and "]" in rag_result["answer"]

        results.append({
            "question": q["question"],
            "context_precision_hit": context_hit,
            "keyword_recall": round(keyword_recall, 2),
            "missing_keywords": [kw for kw in q["expected_keywords"] if kw not in keyword_hits],
            "answer_has_citation": has_citation,
        })

    avg_context_precision = sum(r["context_precision_hit"] for r in results) / len(results)
    avg_keyword_recall = sum(r["keyword_recall"] for r in results) / len(results)

    summary = {
        "num_questions": len(results),
        "avg_context_precision": round(avg_context_precision, 3),
        "avg_keyword_recall": round(avg_keyword_recall, 3),
        "passed": (avg_context_precision >= THRESHOLDS["context_precision"]
                   and avg_keyword_recall >= THRESHOLDS["keyword_recall"]),
        "details": results,
    }
    return summary


if __name__ == "__main__":
    summary = run_eval()
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        print(f"\nEVAL FAILED: below thresholds {THRESHOLDS}")
        sys.exit(1)
    print("\nEVAL PASSED.")