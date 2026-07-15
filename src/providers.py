"""
providers.py

Provider-agnostic LLM interface spanning Anthropic (Claude), OpenAI (GPT-4o),
and AWS Bedrock (Llama 3). Every provider implements the same call_llm()
signature so agent code never needs to know which model answered it.

Each provider falls back to a deterministic mock response if its
credentials aren't configured, so call_provider() is always safe to call —
exactly like src/llm.py's MOCK_MODE, just per-provider instead of global.
"""
import os
import time

PROVIDERS = ["anthropic", "openai", "bedrock_llama"]


def _anthropic_call(system: str, user: str) -> str:
    if os.environ.get("ANTHROPIC_API_KEY") is None:
        return "[MOCK anthropic] Set ANTHROPIC_API_KEY for a real Claude response."
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=500,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _openai_call(system: str, user: str) -> str:
    if os.environ.get("OPENAI_API_KEY") is None:
        return "[MOCK openai] Set OPENAI_API_KEY for a real GPT-4o response."
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def _bedrock_llama_call(system: str, user: str) -> str:
    """
    AWS Bedrock path for Llama 3. Requires AWS credentials with
    bedrock:InvokeModel permission on the target model ARN. Documented and
    unit-testable even without live AWS access — be upfront in interviews
    if you haven't run this against a real account.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") is None:
        return "[MOCK bedrock_llama] Set AWS credentials for a real Llama 3 response."
    import boto3
    import json as _json
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}"
                  f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>",
        "max_gen_len": 500,
    }
    resp = client.invoke_model(
        modelId="meta.llama3-70b-instruct-v1:0",
        body=_json.dumps(body),
    )
    return _json.loads(resp["body"].read())["generation"]


_CALLERS = {
    "anthropic": _anthropic_call,
    "openai": _openai_call,
    "bedrock_llama": _bedrock_llama_call,
}


def call_provider(system: str, user: str, provider: str = "anthropic") -> dict:
    """Returns {"provider", "response", "latency_seconds"} — agents that
    don't care which model answers can call this directly; agents that
    need the mock-mode-aware structured responses (compliance/risk/etc.)
    should keep using src/llm.py's call_llm()."""
    if provider not in _CALLERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from {PROVIDERS}")

    start = time.time()
    response = _CALLERS[provider](system, user)
    latency = round(time.time() - start, 3)
    return {"provider": provider, "response": response, "latency_seconds": latency}


if __name__ == "__main__":
    system = "You are a concise financial research assistant."
    user = "What does 'going concern' mean in a 10-K filing?"
    for provider in PROVIDERS:
        result = call_provider(system, user, provider=provider)
        print(f"[{result['provider']}] ({result['latency_seconds']}s): {result['response'][:120]}")
