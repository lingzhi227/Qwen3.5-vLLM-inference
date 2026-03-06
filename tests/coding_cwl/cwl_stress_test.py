#!/usr/bin/env python3
"""
CWL Agent Stress Test — Run N concurrent CWL agents against vLLM server.
Each agent independently writes CWL files, runs cwltool, debugs errors.
Monitors throughput, latency, and success rate.

Usage:
    python3 cwl_stress_test.py -n 5          # 5 concurrent agents
    python3 cwl_stress_test.py -n 1 3 5 10   # ramp up: test 1, then 3, 5, 10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B")
CWLTOOL_BIN = os.environ.get("CWLTOOL_BIN", "cwltool")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_results")

# Thread-safe print
_print_lock = threading.Lock()
def tprint(agent_id, msg):
    with _print_lock:
        print(f"  [Agent-{agent_id:02d}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tool schemas (same as cwl_agent_test.py)
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
                    "path": {"type": "string", "description": "File path (relative to workspace)"},
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
                    "command": {"type": "string", "description": "The shell command to execute"},
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

SYSTEM_PROMPT = (
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
)


# ---------------------------------------------------------------------------
# Single agent runner (runs in a thread)
# ---------------------------------------------------------------------------
def run_single_agent(agent_id: int, workspace: str) -> dict:
    """Run one CWL agent to completion. Returns result dict."""
    result = {
        "agent_id": agent_id,
        "success": False,
        "rounds": 0,
        "tool_calls": 0,
        "cwltool_runs": 0,
        "duration_s": 0,
        "error": None,
        "first_resp_s": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    t0 = time.time()

    # Prepare workspace
    if os.path.exists(workspace):
        shutil.rmtree(workspace, ignore_errors=True)
    os.makedirs(workspace, exist_ok=True)

    # Tool implementations (workspace-scoped)
    def write_file(path, content):
        if not os.path.isabs(path):
            path = os.path.join(workspace, path)
        if not path.startswith(workspace):
            return {"error": "Path outside workspace"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"status": "ok", "bytes": len(content)}

    def run_command(command):
        command = command.replace("cwltool", CWLTOOL_BIN)
        try:
            r = subprocess.run(command, shell=True, cwd=workspace,
                               capture_output=True, text=True, timeout=60)
            return {"exit_code": r.returncode,
                    "stdout": r.stdout[-3000:] if r.stdout else "",
                    "stderr": r.stderr[-3000:] if r.stderr else ""}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Timeout 60s"}

    def read_file(path):
        if not os.path.isabs(path):
            path = os.path.join(workspace, path)
        if not path.startswith(workspace):
            return {"error": "Path outside workspace"}
        try:
            with open(path) as f:
                return {"status": "ok", "content": f.read()}
        except FileNotFoundError:
            return {"error": f"Not found: {path}"}

    dispatch = {"write_file": write_file, "run_command": run_command, "read_file": read_file}

    try:
        client = OpenAI(base_url=BASE_URL, api_key="empty")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Create the CWL workflow, write all files, and run cwltool. Fix any errors until it succeeds."},
        ]

        tprint(agent_id, "started")

        for round_num in range(1, 21):
            result["rounds"] = round_num
            round_t0 = time.time()

            # Non-streaming call (simpler for concurrent test)
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                max_tokens=8192,
                temperature=0.3,
                top_p=0.9,
                stream=False,
            )

            if result["first_resp_s"] is None:
                result["first_resp_s"] = time.time() - round_t0

            # Track token usage
            if response.usage:
                result["prompt_tokens"] += response.usage.prompt_tokens
                result["completion_tokens"] += response.usage.completion_tokens

            choice = response.choices[0]
            assistant_msg = {"role": "assistant", "content": choice.message.content}

            if choice.message.tool_calls:
                tool_calls = []
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    })
                assistant_msg["tool_calls"] = tool_calls
                if not choice.message.content:
                    assistant_msg["content"] = None
                messages.append(assistant_msg)

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    result["tool_calls"] += 1
                    fn = dispatch.get(fn_name)
                    fn_result = fn(**fn_args) if fn else {"error": f"Unknown: {fn_name}"}
                    fn_result_str = json.dumps(fn_result, ensure_ascii=False)

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": fn_result_str})

                    # Check cwltool success
                    if fn_name == "run_command":
                        result["cwltool_runs"] += 1
                        if (isinstance(fn_result, dict) and fn_result.get("exit_code") == 0
                                and "cwltool" in str(fn_args.get("command", ""))):
                            result["success"] = True
                            tprint(agent_id, f"SUCCESS round {round_num} ({time.time()-t0:.1f}s)")
            else:
                messages.append(assistant_msg)

            if result["success"] or not choice.message.tool_calls:
                break

        if not result["success"]:
            tprint(agent_id, f"FAILED after {round_num} rounds ({time.time()-t0:.1f}s)")

    except Exception as e:
        result["error"] = str(e)
        tprint(agent_id, f"ERROR: {e}")

    result["duration_s"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# GPU memory monitor (background thread, queries via vLLM metrics)
# ---------------------------------------------------------------------------
class GPUMonitor:
    def __init__(self, api_url, interval=3):
        self.api_url = api_url.replace("/v1", "")  # base URL without /v1
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor(self):
        import urllib.request
        while not self._stop.is_set():
            try:
                req = urllib.request.urlopen(f"{self.api_url}/metrics", timeout=5)
                text = req.read().decode()
                sample = {"time": time.time()}
                for line in text.split("\n"):
                    if line.startswith("vllm:kv_cache_usage_perc{"):
                        sample["kv_cache_pct"] = float(line.split()[-1]) * 100
                    elif line.startswith("vllm:num_requests_running{"):
                        sample["requests_running"] = int(float(line.split()[-1]))
                    elif line.startswith("vllm:num_requests_waiting{"):
                        sample["requests_waiting"] = int(float(line.split()[-1]))
                    elif line.startswith("vllm:generation_tokens_total{"):
                        sample["gen_tokens_total"] = float(line.split()[-1])
                    elif line.startswith("vllm:prompt_tokens_total{"):
                        sample["prompt_tokens_total"] = float(line.split()[-1])
                if len(sample) > 1:
                    self.samples.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {"error": "no samples collected"}
        kv = [s.get("kv_cache_pct", 0) for s in self.samples if "kv_cache_pct" in s]
        running = [s.get("requests_running", 0) for s in self.samples if "requests_running" in s]
        waiting = [s.get("requests_waiting", 0) for s in self.samples if "requests_waiting" in s]
        result = {}
        if kv:
            result["kv_cache_pct_max"] = max(kv)
            result["kv_cache_pct_avg"] = sum(kv) / len(kv)
        if running:
            result["max_concurrent"] = max(running)
            result["avg_concurrent"] = sum(running) / len(running)
        if waiting:
            result["max_waiting"] = max(waiting)
        result["samples"] = len(self.samples)
        return result


# ---------------------------------------------------------------------------
# Run one concurrency level
# ---------------------------------------------------------------------------
def run_batch(n_agents: int, gpu_monitor: GPUMonitor = None) -> dict:
    print(f"\n{'='*70}")
    print(f"  STRESS TEST: {n_agents} concurrent CWL agents")
    print(f"  Server: {BASE_URL}")
    print(f"  Model:  {MODEL}")
    print(f"{'='*70}\n")

    # Use port or PID to isolate workspaces when multiple instances run
    ws_suffix = os.environ.get("STRESS_WORKSPACE_ID", str(os.getpid()))
    base_workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"stress_workspaces_{ws_suffix}")
    os.makedirs(base_workspace, exist_ok=True)

    gpu_start_idx = len(gpu_monitor.samples) if gpu_monitor else 0

    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = {}
        for i in range(n_agents):
            ws = os.path.join(base_workspace, f"agent_{i:02d}")
            futures[pool.submit(run_single_agent, i, ws)] = i

        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - t0
    results.sort(key=lambda r: r["agent_id"])

    # GPU stats for this batch
    gpu_stats = None
    if gpu_monitor:
        batch_samples = gpu_monitor.samples[gpu_start_idx:]
        if batch_samples:
            kv = [s.get("kv_cache_pct", 0) for s in batch_samples if "kv_cache_pct" in s]
            running = [s.get("requests_running", 0) for s in batch_samples if "requests_running" in s]
            waiting = [s.get("requests_waiting", 0) for s in batch_samples if "requests_waiting" in s]
            gpu_stats = {}
            if kv:
                gpu_stats["kv_cache_pct_max"] = round(max(kv), 1)
                gpu_stats["kv_cache_pct_avg"] = round(sum(kv) / len(kv), 1)
            if running:
                gpu_stats["max_concurrent"] = max(running)
            if waiting:
                gpu_stats["max_waiting"] = max(waiting)

    # Print results table
    print(f"\n{'─'*90}")
    print(f"  {'Agent':>7} | {'Status':>8} | {'Rounds':>6} | {'Tools':>5} | {'Prompt':>7} | {'Compl':>7} | {'tok/s':>6} | {'Total':>7}")
    print(f"  {'─'*7} | {'─'*8} | {'─'*6} | {'─'*5} | {'─'*7} | {'─'*7} | {'─'*6} | {'─'*7}")
    for r in results:
        if r["error"]:
            status = "\033[31mERR\033[0m "
        elif r["success"]:
            status = "\033[32m OK\033[0m "
        else:
            status = "\033[31mFAIL\033[0m"
        tps = r["completion_tokens"] / r["duration_s"] if r["duration_s"] > 0 else 0
        print(f"  {'#'+str(r['agent_id']):>7} | {status:>17} | {r['rounds']:>6} | {r['tool_calls']:>5} | {r['prompt_tokens']:>7} | {r['completion_tokens']:>7} | {tps:>6.1f} | {r['duration_s']:>6.1f}s")
    print(f"  {'─'*90}")

    n_success = sum(1 for r in results if r["success"])
    n_error = sum(1 for r in results if r["error"])
    avg_time = sum(r["duration_s"] for r in results) / len(results) if results else 0
    durations = [r["duration_s"] for r in results if r["success"]]
    first_resps = [r["first_resp_s"] for r in results if r["first_resp_s"]]
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_tokens = total_prompt + total_completion

    print(f"\n  Summary ({n_agents} agents):")
    print(f"    Success:        {n_success}/{n_agents} ({100*n_success/n_agents:.0f}%)")
    if n_error:
        print(f"    Errors:         {n_error}")
    print(f"    Wall time:      {total_time:.1f}s")
    print(f"    Avg per agent:  {avg_time:.1f}s")
    if durations:
        print(f"    Fastest agent:  {min(durations):.1f}s")
        print(f"    Slowest agent:  {max(durations):.1f}s")
    if first_resps:
        print(f"    Avg 1st resp:   {sum(first_resps)/len(first_resps):.1f}s")
    print(f"\n  Throughput:")
    print(f"    Prompt tokens:  {total_prompt:,}")
    print(f"    Compl tokens:   {total_completion:,}")
    print(f"    Total tokens:   {total_tokens:,}")
    if total_time > 0:
        print(f"    Throughput:     {total_completion/total_time:.1f} gen tok/s  |  {total_tokens/total_time:.1f} total tok/s")

    if gpu_stats:
        print(f"\n  GPU / vLLM:")
        if "kv_cache_pct_max" in gpu_stats:
            print(f"    KV cache peak:  {gpu_stats['kv_cache_pct_max']:.1f}%")
            print(f"    KV cache avg:   {gpu_stats['kv_cache_pct_avg']:.1f}%")
        if "max_concurrent" in gpu_stats:
            print(f"    Max concurrent: {gpu_stats['max_concurrent']}")
        if "max_waiting" in gpu_stats:
            print(f"    Max queued:     {gpu_stats['max_waiting']}")

    return {
        "n_agents": n_agents,
        "n_success": n_success,
        "n_error": n_error,
        "total_time_s": round(total_time, 1),
        "avg_agent_time_s": round(avg_time, 1),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "gen_tok_per_s": round(total_completion / total_time, 1) if total_time > 0 else 0,
        "agents": results,
        "gpu": gpu_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CWL Agent Stress Test")
    parser.add_argument("-n", nargs="+", type=int, default=[5],
                        help="Concurrent agents (multiple values = ramp-up test)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verify server
    try:
        client = OpenAI(base_url=BASE_URL, api_key="empty")
        models = client.models.list()
        print(f"[OK] Server: {models.data[0].id}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to {BASE_URL}: {e}")
        return 1

    # Start GPU monitor via vLLM /metrics endpoint
    gpu_monitor = GPUMonitor(BASE_URL, interval=3)
    gpu_monitor.start()
    print(f"[OK] GPU monitor started (vLLM /metrics)")

    all_results = []
    for n in args.n:
        batch = run_batch(n, gpu_monitor)
        all_results.append(batch)

    gpu_monitor.stop()

    # Overall GPU summary
    gs = gpu_monitor.summary()
    if "error" not in gs:
        print(f"\n{'='*70}")
        print(f"  GPU Monitor Overall ({gs['samples']} samples)")
        if "kv_cache_pct_max" in gs:
            print(f"    KV cache peak:    {gs['kv_cache_pct_max']:.1f}%")
        if "max_concurrent" in gs:
            print(f"    Max concurrent:   {gs['max_concurrent']}")
        if "max_waiting" in gs:
            print(f"    Max queued:       {gs['max_waiting']}")
        print(f"{'='*70}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    levels = "_".join(str(n) for n in args.n)
    result_path = os.path.join(RESULTS_DIR, f"stress_{levels}_{ts}.json")
    with open(result_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "server": BASE_URL,
            "model": MODEL,
            "concurrency_levels": args.n,
            "batches": all_results,
            "gpu_overall": gs,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Ramp-up summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  RAMP-UP SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Agents':>7} | {'Success':>8} | {'Wall(s)':>8} | {'Avg(s)':>7} | {'Gen tok/s':>9} | {'KV Peak':>8} | {'MaxConc':>7}")
        print(f"  {'─'*7} | {'─'*8} | {'─'*8} | {'─'*7} | {'─'*9} | {'─'*8} | {'─'*7}")
        for b in all_results:
            kv = f"{b['gpu']['kv_cache_pct_max']}%" if b.get("gpu") and "kv_cache_pct_max" in b["gpu"] else "N/A"
            mc = str(b["gpu"]["max_concurrent"]) if b.get("gpu") and "max_concurrent" in b["gpu"] else "N/A"
            gen_tps = f"{b.get('gen_tok_per_s', 0):.1f}" if b.get("gen_tok_per_s") else "N/A"
            print(f"  {b['n_agents']:>7} | {b['n_success']}/{b['n_agents']:>5} | {b['total_time_s']:>8} | {b['avg_agent_time_s']:>7} | {gen_tps:>9} | {kv:>8} | {mc:>7}")
        print(f"  {'─'*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
