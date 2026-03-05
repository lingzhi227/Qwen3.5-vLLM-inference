#!/usr/bin/env python3
"""
CWL Stress Test — Run 10 concurrent agent sessions with different CWL tasks.
Each session gets its own workspace, trace file, and unique task.
Tests vLLM's queuing/scheduling under concurrent load.

Optimizations over v1:
- Streaming API so enable_thinking works (thinking tokens don't count toward max_tokens)
- Strip <think> content from message history to prevent context bloat
- Per-round timeout (90s) and per-session timeout (5min)
- Reduced max_tokens to 4096 (sufficient for tool calls)
"""

import json
import os
import re
import subprocess
import sys
import time
import socket
import threading
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-9B"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_CWLTOOL = os.path.join(PROJECT_DIR, "venv", "bin", "cwltool")

MAX_TOKENS = 4096
MAX_ROUNDS = 15
ROUND_TIMEOUT = 90        # seconds per API call
SESSION_TIMEOUT = 300     # 5 min total per session

SSH_TUNNEL_CMD = [
    "ssh", "-p", "2222", "-i", "/Users/lingzhi/.ssh/id_ed25519",
    "-f", "-N", "-L", "8000:localhost:8000",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "lingzhi@108.41.63.249",
]

# 10 different CWL tasks — each agent gets a unique one
TASKS = [
    {
        "id": "wc",
        "desc": "Count lines/words/chars in a file using `wc`",
        "command": "wc",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `wc` on an input text file and captures stdout output.",
    },
    {
        "id": "sort",
        "desc": "Sort lines in a file using `sort`",
        "command": "sort",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `sort` on an input text file and captures the sorted output to stdout.",
    },
    {
        "id": "head",
        "desc": "Get first 3 lines using `head -n 3`",
        "command": "head",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `head -n 3` on an input text file. Use baseCommand: head, and add '-n' and '3' as arguments before the file input.",
    },
    {
        "id": "tail",
        "desc": "Get last 2 lines using `tail -n 2`",
        "command": "tail",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `tail -n 2` on an input text file. Use baseCommand: tail, and add '-n' and '2' as arguments before the file input.",
    },
    {
        "id": "cat",
        "desc": "Concatenate and output file using `cat`",
        "command": "cat",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `cat` on an input text file and captures stdout output.",
    },
    {
        "id": "rev",
        "desc": "Reverse each line using `rev`",
        "command": "rev",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `rev` on an input text file to reverse each line, capturing stdout output.",
    },
    {
        "id": "tr",
        "desc": "Convert lowercase to uppercase using `tr`",
        "command": "tr",
        "prompt": "Create a CWL v1.2 CommandLineTool that converts a text file to uppercase. Use baseCommand: tr, with arguments 'a-z' 'A-Z', and pipe the input file via stdin (set the input type to File and add 'streamable: true' with 'inputBinding: {loadContents: false}', then set 'stdin: $(inputs.input_file.path)' at the top level).",
    },
    {
        "id": "nl",
        "desc": "Number lines using `nl`",
        "command": "nl",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `nl` (number lines) on an input text file and captures the numbered output to stdout.",
    },
    {
        "id": "uniq",
        "desc": "Remove duplicate adjacent lines using `uniq`",
        "command": "uniq",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `uniq` on an input text file to remove duplicate adjacent lines, capturing stdout output.",
    },
    {
        "id": "fold",
        "desc": "Wrap lines at 40 chars using `fold -w 40`",
        "command": "fold",
        "prompt": "Create a CWL v1.2 CommandLineTool that runs `fold -w 40` on an input text file to wrap long lines at 40 characters. Use baseCommand: fold, with '-w' and '40' as arguments.",
    },
]

SYSTEM_PROMPT_TEMPLATE = (
    "You have tools: write_file, run_command, read_file. "
    "Task: {task_prompt} "
    "Write the .cwl file, input .yml, and a sample .txt file. "
    "Run with: cwltool <name>.cwl input.yml. "
    "If errors, read them, fix files, retry until exit code 0. "
    "After success, read and report the output. "
    "Be concise — do not explain at length, just act.\n\n"
    "Here is a minimal CWL v1.2 CommandLineTool example for reference:\n"
    "```\n"
    "cwlVersion: v1.2\n"
    "class: CommandLineTool\n"
    "baseCommand: echo\n"
    "stdout: output.txt\n"
    "inputs:\n"
    "  message:\n"
    "    type: string\n"
    "    inputBinding:\n"
    "      position: 1\n"
    "outputs:\n"
    "  out:\n"
    "    type: stdout\n"
    "```\n"
    "Key points: inputs use inputBinding with position, "
    "stdout capture uses 'type: stdout' in outputs and 'stdout: filename' at top level. "
    "For File inputs, just use 'type: File' with inputBinding position — no valueFrom needed. "
    "For stdin piping, use 'stdin: $(inputs.input_file.path)' at top level (NOT inside inputBinding)."
)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def make_tools(work_dir):
    def write_file(path, content):
        if not os.path.isabs(path):
            path = os.path.join(work_dir, path)
        if not path.startswith(work_dir):
            return {"error": f"Path must be inside workspace: {work_dir}"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "ok", "path": path, "bytes": len(content)}

    def run_command(command):
        command = command.replace("cwltool", VENV_CWLTOOL)
        try:
            result = subprocess.run(
                command, shell=True, cwd=work_dir,
                capture_output=True, text=True, timeout=60,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout[-3000:] if result.stdout else "",
                "stderr": result.stderr[-3000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Command timed out (60s)"}

    def read_file(path):
        if not os.path.isabs(path):
            path = os.path.join(work_dir, path)
        if not path.startswith(work_dir):
            return {"error": f"Path must be inside workspace: {work_dir}"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"status": "ok", "path": path, "content": f.read()}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}

    return {"write_file": write_file, "run_command": run_command, "read_file": read_file}


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "write_file", "description": "Write content to a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def strip_think(text):
    """Remove thinking content from text to keep message history lean.

    vLLM strips the <think> special token but keeps </think> in content,
    so content looks like: 'thinking text...</think>answer text'
    We split on </think> and keep only the answer part.
    """
    if not text:
        return text
    if "</think>" in text:
        # Everything after the last </think> is the actual answer
        parts = text.rsplit("</think>", 1)
        return parts[-1].strip()
    # No </think> found — could be pure answer or pure thinking
    # In later rounds the model may skip thinking entirely
    return text.strip()


# ---------------------------------------------------------------------------
# Streaming helper — collect a full response via streaming API
# ---------------------------------------------------------------------------
def streaming_chat(client, messages, timeout_s=ROUND_TIMEOUT):
    """
    Call chat.completions.create with stream=True and enable_thinking=True.
    Collects the full response. Returns (content, tool_calls_list, finish_reason).
    Raises TimeoutError if it takes too long.
    """
    deadline = time.time() + timeout_s

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    content_parts = []
    tool_calls_map = {}  # index -> {id, name, arguments}
    finish_reason = None

    for chunk in stream:
        if time.time() > deadline:
            stream.close()
            raise TimeoutError(f"Streaming response exceeded {timeout_s}s")

        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            continue

        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        # Content
        if delta.content:
            content_parts.append(delta.content)

        # Tool calls (streamed incrementally)
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": tc_delta.id or "",
                        "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                        "arguments": "",
                    }
                else:
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_map[idx]["name"] = tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tool_calls_map[idx]["arguments"] += tc_delta.function.arguments

    content = "".join(content_parts)
    tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())] if tool_calls_map else []

    return content, tool_calls, finish_reason


# ---------------------------------------------------------------------------
# Single agent session (runs in a thread)
# ---------------------------------------------------------------------------
def run_agent_session(task, results, lock):
    task_id = task["id"]
    work_dir = os.path.join(PROJECT_DIR, "cwl_workspaces", task_id)
    os.makedirs(work_dir, exist_ok=True)
    # Clean workspace
    for f in os.listdir(work_dir):
        fp = os.path.join(work_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    tool_dispatch = make_tools(work_dir)
    client = OpenAI(base_url=BASE_URL, api_key="empty")

    trace_events = []
    start_time = time.time()

    def record(event_type, data):
        trace_events.append({
            "timestamp": time.time() - start_time,
            "type": event_type,
            "data": data,
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(task_prompt=task["prompt"])},
        {"role": "user", "content": f"Create the CWL workflow for `{task['command']}`, write all files, and run cwltool. Fix errors until it succeeds."},
    ]
    record("user_prompt", {"task_id": task_id, "command": task["command"]})

    success = False
    rounds_used = 0
    error_msg = None

    for round_num in range(1, MAX_ROUNDS + 1):
        # Session timeout check
        if time.time() - start_time > SESSION_TIMEOUT:
            record("timeout", {"round": round_num, "reason": "session_timeout", "elapsed": time.time() - start_time})
            error_msg = "session_timeout"
            break

        rounds_used = round_num
        try:
            content, tool_calls, finish_reason = streaming_chat(client, messages)
        except TimeoutError as e:
            record("timeout", {"round": round_num, "reason": "round_timeout", "error": str(e)})
            error_msg = "round_timeout"
            break
        except Exception as e:
            record("error", {"round": round_num, "error": str(e)})
            error_msg = str(e)
            break

        # Strip thinking from content for message history
        clean_content = strip_think(content)

        record("response", {
            "round": round_num,
            "content": clean_content[:300],
            "thinking_chars": len(content) - len(clean_content),
            "tool_calls": [{"name": tc["name"], "args": tc["arguments"][:100]} for tc in tool_calls],
            "finish_reason": finish_reason,
        })

        # Build assistant message with clean content (no thinking)
        assistant_msg = {"role": "assistant", "content": clean_content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["name"]
                try:
                    fn_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                fn = tool_dispatch.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown: {fn_name}"}
                result_str = json.dumps(result, ensure_ascii=False)

                # Check cwltool success
                if (fn_name == "run_command"
                    and isinstance(result, dict)
                    and result.get("exit_code") == 0
                    and "cwltool" in str(fn_args.get("command", ""))):
                    success = True

                record("tool_execution", {
                    "round": round_num,
                    "name": fn_name,
                    "exit_code": result.get("exit_code") if fn_name == "run_command" else None,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            if success:
                # One more round to let model see output and summarize
                pass
        else:
            # No tool calls — model is done talking
            break

    duration = time.time() - start_time
    tool_count = sum(1 for e in trace_events if e["type"] == "tool_execution")
    cwl_runs = sum(1 for e in trace_events if e["type"] == "tool_execution" and e["data"].get("name") == "run_command" and e["data"].get("exit_code") is not None)

    # Save trace
    trace_path = os.path.join(PROJECT_DIR, "cwl_traces", f"trace_{task_id}.json")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL,
            "task_id": task_id,
            "task_command": task["command"],
            "task_desc": task["desc"],
            "created": datetime.now().isoformat(),
            "total_duration_s": duration,
            "success": success,
            "rounds_used": rounds_used,
            "tool_calls": tool_count,
            "cwl_runs": cwl_runs,
            "error": error_msg,
            "events": trace_events,
        }, f, ensure_ascii=False, indent=2)

    # Thread-safe result collection
    with lock:
        results[task_id] = {
            "success": success,
            "rounds": rounds_used,
            "duration": duration,
            "tool_calls": tool_count,
            "cwl_runs": cwl_runs,
            "error": error_msg,
        }
        status = "\033[32mPASS\033[0m" if success else "\033[31mFAIL\033[0m"
        extra = f" [{error_msg}]" if error_msg else ""
        print(f"  [{status}] {task_id:6s} | {rounds_used:2d} rounds | {tool_count:2d} tools | {duration:5.1f}s | {task['desc']}{extra}")


# ---------------------------------------------------------------------------
# SSH tunnel
# ---------------------------------------------------------------------------
def ensure_tunnel():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(2)
        sock.connect(("localhost", 8000))
        sock.close()
        return
    except (ConnectionRefusedError, OSError):
        pass
    subprocess.run(SSH_TUNNEL_CMD, check=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_tunnel()

    client = OpenAI(base_url=BASE_URL, api_key="empty")
    models = client.models.list()
    print(f"[OK] Connected to: {models.data[0].id}")
    print(f"[OK] Launching {len(TASKS)} concurrent agent sessions...\n")

    print(f"{'='*75}")
    print(f"  CONCURRENT CWL STRESS TEST — {len(TASKS)} agents")
    print(f"  max_tokens={MAX_TOKENS}, round_timeout={ROUND_TIMEOUT}s, session_timeout={SESSION_TIMEOUT}s")
    print(f"  streaming=True, enable_thinking=True")
    print(f"{'='*75}")
    print(f"  {'Status':8s} {'Task':6s} | {'Rnds':>4s} | {'Tools':>5s} | {'Time':>6s} | Description")
    print(f"  {'-'*67}")

    results = {}
    lock = threading.Lock()
    threads = []

    wall_start = time.time()

    for task in TASKS:
        t = threading.Thread(target=run_agent_session, args=(task, results, lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    wall_time = time.time() - wall_start

    # Summary
    passed = sum(1 for r in results.values() if r["success"])
    failed = len(results) - passed
    total_tool_calls = sum(r["tool_calls"] for r in results.values())
    sum_duration = sum(r["duration"] for r in results.values())
    timeouts = sum(1 for r in results.values() if r.get("error") and "timeout" in r["error"])

    print(f"\n{'='*75}")
    print(f"  RESULTS")
    print(f"{'='*75}")
    print(f"  Passed:           {passed}/{len(TASKS)}")
    print(f"  Failed:           {failed}/{len(TASKS)}")
    print(f"  Timeouts:         {timeouts}")
    print(f"  Wall-clock time:  {wall_time:.1f}s")
    print(f"  Sum of durations: {sum_duration:.1f}s  (speedup: {sum_duration/wall_time:.1f}x)")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Traces saved to:  {os.path.join(PROJECT_DIR, 'cwl_traces/')}")
    print(f"{'='*75}")

    return 0 if passed == len(TASKS) else 1


if __name__ == "__main__":
    sys.exit(main())
