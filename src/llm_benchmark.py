"""
llm_benchmark.py

Sends the same set of trading-terminology questions to every configured
provider and logs latency + response side by side, so you can make an
evidence-based provider choice instead of a gut-feel one.
"""
import json
from src.providers import call_provider, PROVIDERS

BENCHMARK_QUESTIONS = [
    "What does 'going concern' mean in a 10-K filing?",
    "Explain the difference between a call option and a put option.",
    "What is a CET1 capital ratio and why does it matter for a bank?",
    "What does 'pattern day trader' mean under FINRA rules?",
    "What is portfolio concentration risk?",
]

SYSTEM_PROMPT = "You are a concise financial research assistant. Answer in 2-3 sentences."


def run_benchmark():
    results = []
    for question in BENCHMARK_QUESTIONS:
        for provider in PROVIDERS:
            result = call_provider(SYSTEM_PROMPT, question, provider=provider)
            results.append({"question": question, **result})
            print(f"{provider:15s} | {result['latency_seconds']:.2f}s | {question[:50]}")
    return results


if __name__ == "__main__":
    results = run_benchmark()
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to benchmark_results.json")
