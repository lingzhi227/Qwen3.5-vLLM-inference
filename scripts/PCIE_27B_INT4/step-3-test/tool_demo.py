#!/usr/bin/env python3
"""
Qwen3.5 Tool Calling Demo
- 3 tools: get_weather, calculate, web_search
- Thinking mode enabled
- Streaming output to terminal (same display format as test_cwl_agent.py)
- Full trace captured to JSON + Markdown
- Self-contained: no shell wrapper needed, just `python tool_demo.py`

Env vars:
  VLLM_API_URL  - vLLM API base URL (default: http://localhost:8000/v1)
  VLLM_MODEL    - Model name (auto-detected from server if not set)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "")


# ---------------------------------------------------------------------------
# Markdown trace writer
# ---------------------------------------------------------------------------
class MarkdownTrace:
    """Writes streaming model output to trace.md in real time."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.f.write(f"# Tool Demo Trace\n\n")
        self.f.write(f"- **Model**: {MODEL}\n")
        self.f.write(f"- **Server**: {BASE_URL}\n")
        self.f.write(f"- **Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.f.flush()

    def round_header(self, round_num):
        self.f.write(f"\n---\n\n## Round {round_num}\n\n")
        self.f.flush()

    def thinking(self, text):
        self.f.write(text)
        self.f.flush()

    def thinking_start(self):
        self.f.write("\n### Thinking\n\n```\n")
        self.f.flush()

    def thinking_end(self):
        self.f.write("\n```\n\n")
        self.f.flush()

    def answer_start(self):
        self.f.write("### Response\n\n")
        self.f.flush()

    def answer(self, text):
        self.f.write(text)
        self.f.flush()

    def tool_call(self, name, args_str):
        self.f.write(f"\n### Tool Call: `{name}`\n\n")
        self.f.write(f"```json\n{args_str[:2000]}\n```\n\n")
        self.f.flush()

    def tool_result(self, result_str):
        truncated = result_str[:2000]
        if len(result_str) > 2000:
            truncated += f"\n... (truncated, {len(result_str)} chars total)"
        self.f.write(f"**Result:**\n```\n{truncated}\n```\n\n")
        self.f.flush()

    def summary(self, text):
        self.f.write(f"\n---\n\n## Summary\n\n{text}\n")
        self.f.flush()

    def close(self):
        self.f.close()


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
            {"title": "东京会議場所レンタル", "snippet": "东京国際フォーラム：500平方米，日租金每平方米45元（换算人民币）。品川区域约38元/天。", "url": "https://example.com/tk1"},
            {"title": "东京コンベンション施設ガイド", "snippet": "幕張メッセ大厅1000平方米起租，日租金每平方米30元。交通便利。", "url": "https://example.com/tk2"},
        ],
    }
    for key, results in db.items():
        if key in query:
            return {"query": query, "results": results}
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
# JSON trace recorder
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
            "test": "tool_demo",
            "created": datetime.now().isoformat(),
            "total_duration_s": time.time() - self.start_time,
            "events": self.events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*60}")
        print(f"  JSON trace saved to {path}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Streaming chat loop
# ---------------------------------------------------------------------------
def _execute_tool_call(tc_data, messages, trace, md, round_num, t0):
    """Execute a completed tool call: show args, run, show result."""
    fn_name = tc_data["name"]
    try:
        fn_args = json.loads(tc_data["arguments"])
    except json.JSONDecodeError:
        fn_args = {}

    args_display = json.dumps(fn_args, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"  \033[33m>\033[0m ({elapsed:.2f}s) {fn_name}({args_display[:200]})")
    sys.stdout.flush()
    md.tool_call(fn_name, args_display)

    # Execute
    exec_start = time.time()
    fn = TOOL_DISPATCH.get(fn_name)
    if fn:
        result = fn(**fn_args)
    else:
        result = {"error": f"Unknown tool: {fn_name}"}
    exec_dur = time.time() - exec_start

    result_str = json.dumps(result, ensure_ascii=False)
    elapsed = time.time() - t0
    if len(result_str) > 500:
        print(f"  \033[33m<\033[0m ({elapsed:.2f}s) ({len(result_str)} chars) (exec {exec_dur*1000:.0f}ms)")
    else:
        print(f"  \033[33m<\033[0m ({elapsed:.2f}s) {result_str} (exec {exec_dur*1000:.0f}ms)")
    sys.stdout.flush()

    md.tool_result(result_str)

    trace.record("tool_execution", {
        "round": round_num,
        "tool_call_id": tc_data["id"],
        "name": fn_name,
        "arguments": fn_args,
        "result": result,
    })

    messages.append({
        "role": "tool",
        "tool_call_id": tc_data["id"],
        "content": result_str,
    })

    return {
        "id": tc_data["id"],
        "type": "function",
        "function": {"name": fn_name, "arguments": tc_data["arguments"]},
    }


def stream_chat(client: OpenAI, messages: list, trace: Trace, md: MarkdownTrace, round_num: int):
    print(f"\n{'='*60}")
    print(f"  Round {round_num}")
    print(f"{'='*60}")

    md.round_header(round_num)
    trace.record("request", {"round": round_num, "message_count": len(messages)})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": True},
    )

    # --- Accumulators ---
    raw_content_parts = []
    thinking_parts = []
    answer_parts = []
    tool_calls_acc = {}       # index -> {id, name, arguments}
    tool_calls_flushed = set()  # indices already executed
    tool_calls_list = []      # final list for assistant message
    finish_reason = None

    state = "thinking"
    thinking_header_shown = False
    answer_header_shown = False
    md_thinking_started = False
    md_answer_started = False
    tool_header_shown = False
    usage_info = None
    t0 = time.time()

    def _flush_tool(idx):
        """Execute tool call at given index if not already flushed."""
        nonlocal tool_header_shown
        if idx in tool_calls_flushed:
            return
        tc_data = tool_calls_acc[idx]
        if not tc_data["name"]:
            return
        tool_calls_flushed.add(idx)
        if not tool_header_shown:
            print(f"\n\033[33m{'─'*40}\033[0m")
            print(f"\033[33m  TOOL CALLS\033[0m")
            print(f"\033[33m{'─'*40}\033[0m")
            tool_header_shown = True
        tc_result = _execute_tool_call(tc_data, messages, trace, md, round_num, t0)
        tool_calls_list.append(tc_result)

    for chunk in stream:
        # Usage info comes in the final chunk (no choices)
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason or finish_reason

        if delta.content:
            text = delta.content
            raw_content_parts.append(text)

            if state == "thinking":
                if "</think>" in text:
                    before_end, after_end = text.split("</think>", 1)
                    if before_end:
                        if before_end.strip() and not thinking_header_shown:
                            print(f"\n\033[36m{'─'*40}\033[0m")
                            print(f"\033[36m  THINKING\033[0m")
                            print(f"\033[36m{'─'*40}\033[0m")
                            thinking_header_shown = True
                        if not md_thinking_started:
                            md.thinking_start()
                            md_thinking_started = True
                        thinking_parts.append(before_end)
                        md.thinking(before_end)
                        if thinking_header_shown:
                            sys.stdout.write(f"\033[90m{before_end}\033[0m")
                            sys.stdout.flush()
                    if md_thinking_started:
                        md.thinking_end()
                    state = "answering"
                    after_end = after_end.lstrip("\n")
                    if after_end.strip():
                        print(f"\n\033[32m{'─'*40}\033[0m")
                        print(f"\033[32m  ANSWER\033[0m")
                        print(f"\033[32m{'─'*40}\033[0m")
                        answer_header_shown = True
                        if not md_answer_started:
                            md.answer_start()
                            md_answer_started = True
                        answer_parts.append(after_end)
                        md.answer(after_end)
                        sys.stdout.write(after_end)
                        sys.stdout.flush()
                else:
                    if text.strip() and not thinking_header_shown:
                        print(f"\n\033[36m{'─'*40}\033[0m")
                        print(f"\033[36m  THINKING\033[0m")
                        print(f"\033[36m{'─'*40}\033[0m")
                        thinking_header_shown = True
                    if not md_thinking_started and text.strip():
                        md.thinking_start()
                        md_thinking_started = True
                    thinking_parts.append(text)
                    if md_thinking_started:
                        md.thinking(text)
                    if thinking_header_shown:
                        sys.stdout.write(f"\033[90m{text}\033[0m")
                        sys.stdout.flush()

            elif state == "answering":
                if text.strip() and not answer_header_shown:
                    print(f"\n\033[32m{'─'*40}\033[0m")
                    print(f"\033[32m  ANSWER\033[0m")
                    print(f"\033[32m{'─'*40}\033[0m")
                    answer_header_shown = True
                if not md_answer_started and text.strip():
                    md.answer_start()
                    md_answer_started = True
                answer_parts.append(text)
                if md_answer_started:
                    md.answer(text)
                if answer_header_shown:
                    sys.stdout.write(text)
                    sys.stdout.flush()

        # Tool calls — streamed incrementally
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    # New tool call index appeared — flush the previous one
                    for prev_idx in sorted(tool_calls_acc.keys()):
                        if prev_idx not in tool_calls_flushed:
                            _flush_tool(prev_idx)
                    tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

    # --- Stream ended: flush any remaining unflushed tool calls ---
    for idx in sorted(tool_calls_acc.keys()):
        if idx not in tool_calls_flushed:
            _flush_tool(idx)

    # Close any unclosed md sections
    if md_thinking_started and state == "thinking":
        md.thinking_end()

    # --- Build structured data ---
    full_thinking = "".join(thinking_parts)
    full_answer = "".join(answer_parts)
    full_content = "".join(raw_content_parts)

    # Strip thinking from history to save tokens — only keep the answer
    # The model doesn't need its own thinking in subsequent rounds
    history_content = full_answer if full_answer else full_content
    assistant_msg = {"role": "assistant", "content": history_content}
    if tool_calls_list:
        assistant_msg["tool_calls"] = tool_calls_list
        if not history_content:
            assistant_msg["content"] = None

    trace.record("response", {
        "round": round_num,
        "thinking": full_thinking[:500],
        "answer": full_answer[:1000],
        "tool_calls": [{"name": tc["function"]["name"], "args": tc["function"]["arguments"][:200]} for tc in tool_calls_list],
        "finish_reason": finish_reason,
    })

    messages.append(assistant_msg)

    # --- Usage & cache stats ---
    elapsed = time.time() - t0
    usage_parts = [f"({elapsed:.1f}s)"]
    if usage_info:
        prompt_t = getattr(usage_info, "prompt_tokens", 0)
        compl_t = getattr(usage_info, "completion_tokens", 0)
        usage_parts.append(f"prompt={prompt_t}")
        usage_parts.append(f"completion={compl_t}")
        # vLLM reports cached tokens in prompt_tokens_details
        details = getattr(usage_info, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", None)
            if cached is not None:
                hit_pct = (cached / prompt_t * 100) if prompt_t else 0
                usage_parts.append(f"cache_hit={cached}/{prompt_t} ({hit_pct:.0f}%)")
    print(f"\n  \033[90m[usage] {' | '.join(usage_parts)}\033[0m")

    print()
    return finish_reason, bool(tool_calls_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class TeeWriter:
    """Write to both stdout and a log file, stripping ANSI codes for the file."""
    ANSI_RE = re.compile(r"\033\[[0-9;]*m")

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, text):
        self.terminal.write(text)
        self.log.write(self.ANSI_RE.sub("", text))

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    global BASE_URL, MODEL

    parser = argparse.ArgumentParser(description="Qwen3.5 Tool Calling Demo")
    parser.add_argument("--api-url", default=BASE_URL,
                        help="vLLM API base URL (default: $VLLM_API_URL or http://localhost:8000/v1)")
    parser.add_argument("--model", default=MODEL,
                        help="Model name (default: auto-detect from server)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto-create timestamped dir under tests/)")
    args = parser.parse_args()

    BASE_URL = args.api_url
    MODEL = args.model

    # --- Health check ---
    client = OpenAI(base_url=BASE_URL, api_key="empty")
    try:
        models = client.models.list()
    except Exception as e:
        print(f"\033[31mERROR: vLLM not responding at {BASE_URL}\033[0m")
        print(f"  {e}")
        print(f"\nMake sure vLLM server is running and accessible.")
        return 1

    # Auto-detect model if not specified
    if not MODEL:
        MODEL = models.data[0].id

    # --- Output directory ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        tests_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(tests_base, f"{timestamp}_tool_demo")
    os.makedirs(output_dir, exist_ok=True)

    # --- Console log (tee to file) ---
    tee = TeeWriter(os.path.join(output_dir, "console.log"))
    sys.stdout = tee

    print(f"\033[36m==========================================")
    print(f"  Tool Demo — {MODEL}")
    print(f"  vLLM:   {BASE_URL}")
    print(f"  Output: {output_dir}")
    print(f"==========================================\033[0m")
    print()

    trace = Trace()
    md = MarkdownTrace(os.path.join(output_dir, "trace.md"))

    print(f"[OK] Connected to: {MODEL}")

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
    round_num = 0
    for round_num in range(1, max_rounds + 1):
        finish_reason, had_tools = stream_chat(client, messages, trace, md, round_num)
        if not had_tools:
            break

    duration = time.time() - trace.start_time

    # Save JSON trace
    trace.save(os.path.join(output_dir, "tool_demo_trace.json"))

    # Summary
    tool_calls = [e for e in trace.events if e["type"] == "tool_execution"]
    tool_counts = {}
    for e in tool_calls:
        name = e["data"].get("name", "unknown")
        tool_counts[name] = tool_counts.get(name, 0) + 1

    summary_text = (
        f"- **Rounds**: {round_num}/{max_rounds}\n"
        f"- **Duration**: {duration:.1f}s\n"
        f"- **Total tool calls**: {len(tool_calls)}\n"
    )
    for name, count in tool_counts.items():
        summary_text += f"- **{name}**: {count}\n"
    md.summary(summary_text)
    md.close()

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Rounds: {round_num}/{max_rounds}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total tool calls: {len(tool_calls)}")
    for name, count in tool_counts.items():
        print(f"  - {name}: {count}")
    print(f"\n  Artifacts in: {output_dir}")
    print(f"  - trace.md            (model streaming output)")
    print(f"  - tool_demo_trace.json (structured trace)")
    print(f"  - console.log         (terminal output, no ANSI)")
    print(f"{'='*60}")

    # Restore stdout and close log
    sys.stdout = tee.terminal
    tee.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
