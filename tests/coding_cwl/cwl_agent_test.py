#!/usr/bin/env python3
"""
CWL Agent Test — Test Qwen3.5's agentic coding ability
The model gets 3 terminal tools (write_file, run_command, read_file) and must:
1. Write a valid CWL workflow + input YAML
2. Run cwltool
3. Read error output, debug, fix, and retry
4. Keep iterating until the workflow runs successfully
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B")
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwl_workspace")
CWLTOOL_BIN = os.environ.get("CWLTOOL_BIN", "cwltool")

# ---------------------------------------------------------------------------
# Tool implementations — real terminal interaction
# ---------------------------------------------------------------------------
def write_file(path: str, content: str) -> dict:
    """Write content to a file in the workspace."""
    # Sandbox: force files into WORK_DIR
    if not os.path.isabs(path):
        path = os.path.join(WORK_DIR, path)
    if not path.startswith(WORK_DIR):
        return {"error": f"Path must be inside workspace: {WORK_DIR}"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "path": path, "bytes": len(content)}


def run_command(command: str) -> dict:
    """Run a shell command in the workspace directory."""
    # Replace bare 'cwltool' with our venv path
    command = command.replace("cwltool", CWLTOOL_BIN)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = {
            "exit_code": result.returncode,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
        }
        # Print live feedback
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        print(f"   [{status}] exit={result.returncode}")
        if result.stdout:
            print(f"   stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"   stderr: {result.stderr[:500]}")
        return output
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "Command timed out (60s)"}


def read_file(path: str) -> dict:
    """Read a file from the workspace."""
    if not os.path.isabs(path):
        path = os.path.join(WORK_DIR, path)
    if not path.startswith(WORK_DIR):
        return {"error": f"Path must be inside workspace: {WORK_DIR}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"status": "ok", "path": path, "content": content}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}


TOOL_DISPATCH = {
    "write_file": write_file,
    "run_command": run_command,
    "read_file": read_file,
}

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Path is relative to the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to workspace), e.g. 'hello.cwl'"},
                    "content": {"type": "string", "description": "The full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace directory. Use this to run cwltool, list files, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute, e.g. 'cwltool hello.cwl input.yml'"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to workspace)"},
                },
                "required": ["path"],
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
            "test": "cwl_agent",
            "created": datetime.now().isoformat(),
            "total_duration_s": time.time() - self.start_time,
            "events": self.events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*60}")
        print(f"  Trace saved to {path}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Streaming chat loop (same pattern as tool_demo.py)
# ---------------------------------------------------------------------------
def stream_chat(client: OpenAI, messages: list, trace: Trace, round_num: int):
    print(f"\n{'='*60}")
    print(f"  Round {round_num}")
    print(f"{'='*60}")

    trace.record("request", {"round": round_num, "message_count": len(messages)})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=8192,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    # --- Accumulators ---
    raw_content_parts = []    # everything in delta.content (includes <think> tags)
    thinking_parts = []       # extracted thinking text
    answer_parts = []         # extracted answer text (after </think>)
    tool_calls_acc = {}
    finish_reason = None

    # --- Streaming state machine ---
    # vLLM strips the <think> special token but keeps </think> in content.
    # So: content starts in thinking mode, </think> marks the transition.
    # States: "thinking" -> "answering"
    state = "thinking"
    thinking_header_shown = False
    answer_header_shown = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason or finish_reason

        if delta.content:
            text = delta.content
            raw_content_parts.append(text)

            if state == "thinking":
                if "</think>" in text:
                    # Split at </think> — before is last thinking, after is answer
                    before_end, after_end = text.split("</think>", 1)
                    if before_end:
                        if not thinking_header_shown:
                            print(f"\n\033[36m{'─'*40}\033[0m")
                            print(f"\033[36m  THINKING\033[0m")
                            print(f"\033[36m{'─'*40}\033[0m")
                            thinking_header_shown = True
                        thinking_parts.append(before_end)
                        sys.stdout.write(f"\033[90m{before_end}\033[0m")
                        sys.stdout.flush()
                    state = "answering"
                    if after_end.strip():
                        print(f"\n\033[32m{'─'*40}\033[0m")
                        print(f"\033[32m  ANSWER\033[0m")
                        print(f"\033[32m{'─'*40}\033[0m")
                        answer_header_shown = True
                        answer_parts.append(after_end)
                        sys.stdout.write(after_end)
                        sys.stdout.flush()
                else:
                    # Still thinking — show header on first content
                    if text.strip() and not thinking_header_shown:
                        print(f"\n\033[36m{'─'*40}\033[0m")
                        print(f"\033[36m  THINKING\033[0m")
                        print(f"\033[36m{'─'*40}\033[0m")
                        thinking_header_shown = True
                    thinking_parts.append(text)
                    sys.stdout.write(f"\033[90m{text}\033[0m")
                    sys.stdout.flush()

            elif state == "answering":
                if text.strip() and not answer_header_shown:
                    print(f"\n\033[32m{'─'*40}\033[0m")
                    print(f"\033[32m  ANSWER\033[0m")
                    print(f"\033[32m{'─'*40}\033[0m")
                    answer_header_shown = True
                answer_parts.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

        # Tool calls come via separate delta field
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

    # --- Build structured data ---
    full_thinking = "".join(thinking_parts)
    full_answer = "".join(answer_parts)
    full_content = "".join(raw_content_parts)

    # Use answer only (strip thinking) for message history to save context
    clean_content = full_answer if full_answer else full_content
    assistant_msg = {"role": "assistant", "content": clean_content}
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
        "thinking": full_thinking[:500],
        "answer": full_answer[:1000],
        "tool_calls": [{"name": tc["function"]["name"], "args": tc["function"]["arguments"][:200]} for tc in tool_calls_list],
        "finish_reason": finish_reason,
    })

    messages.append(assistant_msg)

    # --- Execute and display tool calls ---
    if tool_calls_list:
        print(f"\n\033[33m{'─'*40}\033[0m")
        print(f"\033[33m  TOOL CALLS ({len(tool_calls_list)})\033[0m")
        print(f"\033[33m{'─'*40}\033[0m")
        for tc in tool_calls_list:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            print(f"  \033[33m>\033[0m {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")

            fn = TOOL_DISPATCH.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 500:
                print(f"  \033[33m<\033[0m ({len(result_str)} chars)")
            else:
                print(f"  \033[33m<\033[0m {result_str}")

            trace.record("tool_execution", {
                "round": round_num,
                "tool_call_id": tc["id"],
                "name": fn_name,
                "arguments": fn_args if fn_name != "write_file" else {**fn_args, "content": fn_args.get("content", "")[:200] + "..."},
                "result": result if len(result_str) < 2000 else {"status": result.get("status", "ok"), "truncated": True},
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

    # Prepare workspace
    os.makedirs(WORK_DIR, exist_ok=True)
    # Clean previous runs
    for f in os.listdir(WORK_DIR):
        fp = os.path.join(WORK_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    client = OpenAI(base_url=BASE_URL, api_key="empty")
    trace = Trace()

    # Verify server
    models = client.models.list()
    print(f"[OK] Connected to: {models.data[0].id}")

    messages = [
        {
            "role": "system",
            "content": (
                "You have tools: write_file, run_command, read_file. "
                "Task: Create a CWL v1.2 CommandLineTool that runs `wc` on an input file and captures output. "
                "Write the .cwl file, input .yml, and a sample .txt file. "
                "Run with: cwltool wc.cwl input.yml. "
                "If errors, read them, fix files, retry until exit code 0. "
                "After success, read and report the output.\n\n"
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
                "And its input YAML:\n"
                "```\n"
                "message: hello world\n"
                "```\n"
                "Key points: inputs use inputBinding with position, "
                "stdout capture uses 'type: stdout' in outputs and 'stdout: filename' at top level. "
                "For File inputs, just use 'type: File' with inputBinding position — no valueFrom needed."
            ),
        },
        {
            "role": "user",
            "content": "Create the CWL workflow, write all files, and run cwltool. Fix any errors until it succeeds.",
        },
    ]

    trace.record("user_prompt", {"content": messages[-1]["content"]})

    max_rounds = 20
    success = False
    for round_num in range(1, max_rounds + 1):
        finish_reason, had_tools = stream_chat(client, messages, trace, round_num)

        # Check if cwltool succeeded in this round
        for event in trace.events:
            if (event["type"] == "tool_execution"
                and event["data"].get("name") == "run_command"
                and isinstance(event["data"].get("result"), dict)
                and event["data"]["result"].get("exit_code") == 0
                and "cwltool" in str(event["data"].get("arguments", {}).get("command", ""))):
                success = True

        if not had_tools:
            break

    print(f"\n{'='*60}")
    if success:
        print("  RESULT: CWL workflow ran successfully!")
    else:
        print("  RESULT: CWL workflow did NOT succeed within max rounds")
    print(f"  Rounds used: {round_num}/{max_rounds}")
    print(f"{'='*60}")

    # Save trace
    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwl_trace.json")
    trace.save(trace_path)

    # Summary
    tool_calls = [e for e in trace.events if e["type"] == "tool_execution"]
    cwl_runs = [e for e in tool_calls if e["data"].get("name") == "run_command"
                and "cwltool" in str(e["data"].get("arguments", {}).get("command", ""))]
    file_writes = [e for e in tool_calls if e["data"].get("name") == "write_file"]

    print(f"\n  Summary:")
    print(f"  - Total tool calls: {len(tool_calls)}")
    print(f"  - cwltool runs: {len(cwl_runs)}")
    print(f"  - File writes: {len(file_writes)}")
    print(f"  - Successful: {success}")
    print(f"  - Duration: {trace.events[-1]['timestamp']:.1f}s")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
