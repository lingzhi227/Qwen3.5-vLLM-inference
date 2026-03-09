#!/bin/bash
# =============================================================================
# Quick test: verify vLLM Qwen3.5-9B server works (basic + tool calling)
# Usage: bash scripts/PCIE/test.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_pcie"

# Get API URL
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi
API_URL="${API_URL:-http://localhost:8000/v1}"

echo "Testing vLLM server at: $API_URL"

# Health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

MODEL_NAME=$(curl -s "${API_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: $MODEL_NAME"
echo ""

python3 << 'PYEOF'
import os, sys, time, json
from openai import OpenAI

API_URL = os.environ.get("API_URL", "http://localhost:8000/v1")
client = OpenAI(base_url=API_URL, api_key="empty")
model = client.models.list().data[0].id

print("=" * 60)
print("  Test 1: Basic completion")
print("=" * 60)

start = time.time()
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
    max_tokens=64,
    temperature=0.1,
    extra_body={"enable_thinking": True},
)
elapsed = time.time() - start
content = resp.choices[0].message.content
tokens = resp.usage.completion_tokens if resp.usage else 0
print(f"  Response: {content[:200]}")
print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tokens/elapsed:.1f} tok/s")
print()

print("=" * 60)
print("  Test 2: Streaming with thinking")
print("=" * 60)

start = time.time()
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain why the sky is blue in 2 sentences."}],
    max_tokens=512,
    temperature=0.3,
    stream=True,
    extra_body={"enable_thinking": True},
)
tokens = 0
content = []
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        content.append(chunk.choices[0].delta.content)
        tokens += 1
elapsed = time.time() - start
text = "".join(content)
# Strip thinking
if "</think>" in text:
    text = text.split("</think>", 1)[1].strip()
print(f"  Response: {text[:300]}")
print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tokens/elapsed:.1f} tok/s")
print()

print("=" * 60)
print("  Test 3: Tool calling")
print("=" * 60)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]

start = time.time()
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    max_tokens=256,
    temperature=0.1,
    extra_body={"enable_thinking": True},
)
elapsed = time.time() - start
msg = resp.choices[0].message

if msg.tool_calls:
    tc = msg.tool_calls[0]
    print(f"  Tool call: {tc.function.name}({tc.function.arguments})")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  PASS: Tool calling works!")
else:
    print(f"  No tool call returned. Content: {msg.content[:200]}")
    print(f"  WARN: Tool calling may not be working correctly")
print()

print("=" * 60)
print("  All tests complete!")
print("=" * 60)
PYEOF
