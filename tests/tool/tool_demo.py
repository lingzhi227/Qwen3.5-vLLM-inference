#!/usr/bin/env python3
"""
Qwen3.5 Tool Calling Demo
- 3 tools: get_weather, calculate, web_search
- Thinking mode enabled
- Streaming output to terminal
- Full trace captured to JSON

Env vars:
  VLLM_API_URL  - vLLM API base URL (default: http://localhost:8000/v1)
  VLLM_MODEL    - Model name (default: Qwen/Qwen3.5-27B)
"""

import json
import os
import time
import subprocess
import sys
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config (overridable via env vars)
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B")

# ---------------------------------------------------------------------------
# Tool implementations (fake but realistic)
# ---------------------------------------------------------------------------
def get_weather(city: str) -> dict:
    """Simulate weather API."""
    data = {
        "北京": {"temp": -2, "condition": "晴", "humidity": 23, "wind": "北风3级"},
        "上海": {"temp": 8, "condition": "多云", "humidity": 65, "wind": "东风2级"},
        "东京": {"temp": 5, "condition": "小雨", "humidity": 78, "wind": "南风1级"},
    }
    result = data.get(city, {"temp": 15, "condition": "晴", "humidity": 50, "wind": "微风"})
    result["city"] = city
    result["timestamp"] = datetime.now().isoformat()
    return result


def calculate(expression: str) -> dict:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid expression", "expression": expression}
    try:
        result = eval(expression)  # safe: only digits and operators
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def web_search(query: str) -> dict:
    """Simulate web search results with context-dependent data."""
    db = {
        "上海": [
            {"title": "上海会议场地推荐 2026", "snippet": "上海国际会议中心：大厅800平方米，日租金每平方米28元；浦东香格里拉：500平方米，日租金每平方米35元。", "url": "https://example.com/sh1"},
            {"title": "上海团建场地攻略", "snippet": "外滩附近场地均价每平方米30元/天，陆家嘴区域约32元/天。", "url": "https://example.com/sh2"},
        ],
        "北京": [
            {"title": "北京会议场地价格一览", "snippet": "国贸区域大型场地日租金每平方米38元，中关村区域约25元/天，亦庄约18元/天。", "url": "https://example.com/bj1"},
            {"title": "北京年会场地推荐", "snippet": "北京饭店可容纳300人，日租金约45000元。798艺术区有创意空间可租。", "url": "https://example.com/bj2"},
        ],
        "东京": [
            {"title": "东京会议場所レンタル", "snippet": "东京国際フォーラム：500平方米，日租金每平方米45元（换算人民币）。品川区域约38元/天。", "url": "https://example.com/tk1"},
            {"title": "东京コンベンション施設ガイド", "snippet": "幕張メッセ大厅1000平方米起租，日租金每平方米30元。交通便利。", "url": "https://example.com/tk2"},
        ],
    }
    # Match by keyword
    for key, results in db.items():
        if key in query:
            return {"query": query, "results": results}
    # Fallback
    return {
        "query": query,
        "results": [
            {"title": f"关于「{query}」的搜索结果", "snippet": f"2026年最新资料显示，{query}相关信息较多，建议缩小搜索范围。", "url": "https://example.com/1"},
        ],
    }


TOOL_DISPATCH = {
    "get_weather": get_weather,
    "calculate": calculate,
    "web_search": web_search,
}

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如「北京」「上海」"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '(3+5)*2'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------
class Trace:
    def __init__(self):
        self.events = []
        self.start_time = time.time()

    def record(self, event_type: str, data: dict):
        self.events.append({
            "timestamp": time.time() - self.start_time,
            "type": event_type,
            "data": data,
        })

    def save(self, path: str):
        output = {
            "model": MODEL,
            "created": datetime.now().isoformat(),
            "total_duration_s": time.time() - self.start_time,
            "events": self.events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Trace saved to {path}")


# ---------------------------------------------------------------------------
# Streaming chat with tool call handling
# ---------------------------------------------------------------------------
def stream_chat(client: OpenAI, messages: list, trace: Trace, round_num: int):
    """Send a streaming chat request, print output, handle tool calls."""
    print(f"\n{'='*60}")
    print(f"  Round {round_num}")
    print(f"{'='*60}")

    trace.record("request", {
        "round": round_num,
        "messages": [
            {**m, "content": m.get("content", "")[:200]} for m in messages
        ],
    })

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    # Accumulators
    content_parts = []
    tool_calls_acc = {}  # index -> {id, name, arguments}
    finish_reason = None
    in_thinking = False
    usage_info = {}

    for chunk in stream:
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            continue

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason or finish_reason

        # Content (includes thinking)
        if delta.content:
            text = delta.content
            content_parts.append(text)

            # Detect thinking tags
            joined = "".join(content_parts)
            if "<think>" in joined and not in_thinking:
                in_thinking = True
                print("\n🧠 [Thinking]", flush=True)
                # Print what's after <think>
                after = joined.split("<think>", 1)[1]
                if after:
                    sys.stdout.write(f"\033[90m{after}\033[0m")
                    sys.stdout.flush()
            elif "</think>" in text:
                in_thinking = False
                before_end = text.split("</think>")[0]
                after_end = text.split("</think>", 1)[1] if "</think>" in text else ""
                if before_end:
                    sys.stdout.write(f"\033[90m{before_end}\033[0m")
                print("\n\n💬 [Response]", flush=True)
                if after_end:
                    sys.stdout.write(after_end)
                    sys.stdout.flush()
            elif in_thinking:
                sys.stdout.write(f"\033[90m{text}\033[0m")
                sys.stdout.flush()
            else:
                # Only print if we're past thinking or no thinking
                if not any("<think>" in p for p in content_parts[:-1]) or not in_thinking:
                    if "<think>" not in text:
                        sys.stdout.write(text)
                        sys.stdout.flush()

        # Tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc.id or "",
                        "name": tc.function.name or "" if tc.function else "",
                        "arguments": "",
                    }
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

    full_content = "".join(content_parts)

    # Build assistant message for history
    assistant_msg = {"role": "assistant", "content": full_content}
    tool_calls_list = []
    if tool_calls_acc:
        tool_calls_list = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls_acc.values()
        ]
        assistant_msg["tool_calls"] = tool_calls_list
        if not full_content:
            assistant_msg["content"] = None

    trace.record("response", {
        "round": round_num,
        "content": full_content[:500],
        "tool_calls": tool_calls_list,
        "finish_reason": finish_reason,
        "usage": usage_info,
    })

    messages.append(assistant_msg)

    # Execute tool calls
    if tool_calls_list:
        print(f"\n\n🔧 [Tool Calls: {len(tool_calls_list)}]")
        for tc in tool_calls_list:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            print(f"   → {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

            # Dispatch
            fn = TOOL_DISPATCH.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            result_str = json.dumps(result, ensure_ascii=False)
            print(f"   ← {result_str[:200]}")

            trace.record("tool_execution", {
                "round": round_num,
                "tool_call_id": tc["id"],
                "name": fn_name,
                "arguments": fn_args,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str,
            })

    print()
    return finish_reason, bool(tool_calls_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Connecting to: {BASE_URL}")
    client = OpenAI(base_url=BASE_URL, api_key="empty")
    trace = Trace()

    # Verify server
    models = client.models.list()
    print(f"📡 Connected to: {models.data[0].id}")

    # System prompt: force step-by-step, no shortcuts
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous planning assistant. You MUST work step by step:\n"
                "1. First gather all needed data using tools.\n"
                "2. Analyze the results, then decide what additional information is needed.\n"
                "3. Call more tools based on your analysis.\n"
                "4. Only after ALL data is collected, do the final calculations and give your answer.\n"
                "NEVER skip steps. NEVER guess data you haven't retrieved."
            ),
        },
        {
            "role": "user",
            "content": (
                "我在规划公司50人年会，需要你帮我做选址分析：\n\n"
                "第一步：查询北京、上海、东京三个城市的天气，选出温度最接近10°C的城市作为候选。\n\n"
                "第二步：搜索候选城市的会议场地租金信息。\n\n"
                "第三步：根据搜索到的实际租金数据，计算总预算：\n"
                "- 场地：200平方米，租3天（用你搜到的每平方米日租金中最便宜的那个价格）\n"
                "- 机票：50人往返，每人票价 = 该城市温度的绝对值 × 500 + 1000 元\n"
                "- 餐饮：50人 × 3天 × 150元/人/天\n"
                "分别算出三项费用，再算总预算。\n\n"
                "最后给出完整的选址报告。"
            ),
        },
    ]

    trace.record("user_prompt", {"content": messages[-1]["content"]})

    # Agentic loop: keep going until model stops calling tools
    max_rounds = 10
    tool_call_count = 0
    for round_num in range(1, max_rounds + 1):
        finish_reason, had_tools = stream_chat(client, messages, trace, round_num)
        if had_tools:
            tool_call_count += sum(
                1 for m in messages if m["role"] == "tool"
                and not any(
                    e["type"] == "tool_counted" and e["data"]["id"] == m.get("tool_call_id")
                    for e in trace.events
                )
            )
        if not had_tools:
            # Model gave a final answer
            break

    print(f"\n{'='*60}")
    print(f"  Done! Tool calls executed: {tool_call_count}")
    print(f"{'='*60}")

    # Save trace
    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.json")
    trace.save(trace_path)


if __name__ == "__main__":
    main()
