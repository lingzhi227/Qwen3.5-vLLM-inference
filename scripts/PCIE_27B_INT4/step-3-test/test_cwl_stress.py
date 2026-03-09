#!/usr/bin/env python3
"""
CWL Agent Stress Test — Run N concurrent CWL agents against vLLM server.
All config via CLI args, no env vars, no hardcoded paths.

Usage:
    python3 cwl_stress_test.py \
        --api-url http://host:8000/v1 \
        --model model_name \
        --cwltool /path/to/cwltool \
        --output-dir /path/to/output \
        -n 5 10 20
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

# Thread-safe print
_print_lock = threading.Lock()
def tprint(agent_id, msg):
    with _print_lock:
        print(f"  [Agent-{agent_id:02d}] {msg}", flush=True)


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
# Single agent runner
# ---------------------------------------------------------------------------
def run_single_agent(agent_id, workspace, api_url, model, cwltool_bin):
    result = {
        "agent_id": agent_id, "success": False, "rounds": 0,
        "tool_calls": 0, "cwltool_runs": 0, "duration_s": 0,
        "error": None, "first_resp_s": None,
        "prompt_tokens": 0, "completion_tokens": 0,
    }
    t0 = time.time()

    if os.path.exists(workspace):
        shutil.rmtree(workspace, ignore_errors=True)
    os.makedirs(workspace, exist_ok=True)

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
        command = command.replace("cwltool", cwltool_bin)
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
        client = OpenAI(base_url=api_url, api_key="empty")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Create the CWL workflow, write all files, and run cwltool. Fix any errors until it succeeds."},
        ]
        tprint(agent_id, "started")

        for round_num in range(1, 21):
            result["rounds"] = round_num
            round_t0 = time.time()

            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS,
                max_tokens=8192, temperature=0.3, top_p=0.9, stream=False,
            )
            if result["first_resp_s"] is None:
                result["first_resp_s"] = time.time() - round_t0
            if response.usage:
                result["prompt_tokens"] += response.usage.prompt_tokens
                result["completion_tokens"] += response.usage.completion_tokens

            choice = response.choices[0]
            assistant_msg = {"role": "assistant", "content": choice.message.content}

            if choice.message.tool_calls:
                tool_calls = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in choice.message.tool_calls
                ]
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
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": json.dumps(fn_result, ensure_ascii=False)})
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
# GPU monitor
# ---------------------------------------------------------------------------
class GPUMonitor:
    def __init__(self, api_url, interval=3):
        self.api_url = api_url.replace("/v1", "")
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
                if len(sample) > 1:
                    self.samples.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {"error": "no samples collected"}
        kv = [s["kv_cache_pct"] for s in self.samples if "kv_cache_pct" in s]
        running = [s["requests_running"] for s in self.samples if "requests_running" in s]
        waiting = [s["requests_waiting"] for s in self.samples if "requests_waiting" in s]
        r = {"samples": len(self.samples)}
        if kv:
            r["kv_cache_pct_max"] = max(kv)
            r["kv_cache_pct_avg"] = sum(kv) / len(kv)
        if running:
            r["max_concurrent"] = max(running)
        if waiting:
            r["max_waiting"] = max(waiting)
        return r


# ---------------------------------------------------------------------------
# Run one concurrency level
# ---------------------------------------------------------------------------
def run_batch(n_agents, args, gpu_monitor=None):
    print(f"\n{'='*70}")
    print(f"  STRESS TEST: {n_agents} concurrent CWL agents")
    print(f"  Server: {args.api_url}")
    print(f"  Model:  {args.model}")
    print(f"{'='*70}\n")

    ws_dir = os.path.join(args.output_dir, "workspaces")
    os.makedirs(ws_dir, exist_ok=True)

    gpu_start_idx = len(gpu_monitor.samples) if gpu_monitor else 0
    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = {
            pool.submit(run_single_agent, i,
                        os.path.join(ws_dir, f"agent_{i:02d}"),
                        args.api_url, args.model, args.cwltool): i
            for i in range(n_agents)
        }
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - t0
    results.sort(key=lambda r: r["agent_id"])

    # GPU stats
    gpu_stats = None
    if gpu_monitor:
        batch_samples = gpu_monitor.samples[gpu_start_idx:]
        if batch_samples:
            kv = [s["kv_cache_pct"] for s in batch_samples if "kv_cache_pct" in s]
            running = [s["requests_running"] for s in batch_samples if "requests_running" in s]
            waiting = [s["requests_waiting"] for s in batch_samples if "requests_waiting" in s]
            gpu_stats = {}
            if kv:
                gpu_stats["kv_cache_pct_max"] = round(max(kv), 1)
                gpu_stats["kv_cache_pct_avg"] = round(sum(kv) / len(kv), 1)
            if running:
                gpu_stats["max_concurrent"] = max(running)
            if waiting:
                gpu_stats["max_waiting"] = max(waiting)

    # Print table
    print(f"\n{'─'*90}")
    print(f"  {'Agent':>7} | {'Status':>8} | {'Rounds':>6} | {'Tools':>5} | {'Prompt':>7} | {'Compl':>7} | {'tok/s':>6} | {'Total':>7}")
    print(f"  {'─'*7} | {'─'*8} | {'─'*6} | {'─'*5} | {'─'*7} | {'─'*7} | {'─'*6} | {'─'*7}")
    for r in results:
        st = "\033[31mERR\033[0m " if r["error"] else ("\033[32m OK\033[0m " if r["success"] else "\033[31mFAIL\033[0m")
        tps = r["completion_tokens"] / r["duration_s"] if r["duration_s"] > 0 else 0
        print(f"  {'#'+str(r['agent_id']):>7} | {st:>17} | {r['rounds']:>6} | {r['tool_calls']:>5} | {r['prompt_tokens']:>7} | {r['completion_tokens']:>7} | {tps:>6.1f} | {r['duration_s']:>6.1f}s")
    print(f"  {'─'*90}")

    n_ok = sum(1 for r in results if r["success"])
    n_err = sum(1 for r in results if r["error"])
    avg_t = sum(r["duration_s"] for r in results) / len(results)
    durs = [r["duration_s"] for r in results if r["success"]]
    fr = [r["first_resp_s"] for r in results if r["first_resp_s"]]
    tp = sum(r["prompt_tokens"] for r in results)
    tc = sum(r["completion_tokens"] for r in results)

    print(f"\n  Summary ({n_agents} agents):")
    print(f"    Success:        {n_ok}/{n_agents} ({100*n_ok/n_agents:.0f}%)")
    if n_err: print(f"    Errors:         {n_err}")
    print(f"    Wall time:      {total_time:.1f}s")
    print(f"    Avg per agent:  {avg_t:.1f}s")
    if durs:
        print(f"    Fastest agent:  {min(durs):.1f}s")
        print(f"    Slowest agent:  {max(durs):.1f}s")
    if fr: print(f"    Avg 1st resp:   {sum(fr)/len(fr):.1f}s")
    print(f"\n  Throughput:")
    print(f"    Prompt tokens:  {tp:,}")
    print(f"    Compl tokens:   {tc:,}")
    print(f"    Total tokens:   {tp+tc:,}")
    if total_time > 0:
        print(f"    Throughput:     {tc/total_time:.1f} gen tok/s  |  {(tp+tc)/total_time:.1f} total tok/s")
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
        "n_agents": n_agents, "n_success": n_ok, "n_error": n_err,
        "total_time_s": round(total_time, 1), "avg_agent_time_s": round(avg_t, 1),
        "total_prompt_tokens": tp, "total_completion_tokens": tc,
        "gen_tok_per_s": round(tc / total_time, 1) if total_time > 0 else 0,
        "agents": results, "gpu": gpu_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CWL Agent Stress Test")
    p.add_argument("--api-url", required=True, help="vLLM API URL")
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument("--cwltool", required=True, help="Path to cwltool binary")
    p.add_argument("--output-dir", required=True, help="Output directory for results and workspaces")
    p.add_argument("-n", nargs="+", type=int, default=[5], help="Concurrent agents (multiple = ramp-up)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        client = OpenAI(base_url=args.api_url, api_key="empty")
        models = client.models.list()
        print(f"[OK] Server: {models.data[0].id}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to {args.api_url}: {e}")
        return 1

    gpu_monitor = GPUMonitor(args.api_url, interval=3)
    gpu_monitor.start()
    print(f"[OK] GPU monitor started")

    all_results = []
    for n in args.n:
        all_results.append(run_batch(n, args, gpu_monitor))

    gpu_monitor.stop()

    gs = gpu_monitor.summary()
    if "error" not in gs:
        print(f"\n{'='*70}")
        print(f"  GPU Monitor Overall ({gs['samples']} samples)")
        if "kv_cache_pct_max" in gs: print(f"    KV cache peak:    {gs['kv_cache_pct_max']:.1f}%")
        if "max_concurrent" in gs:   print(f"    Max concurrent:   {gs['max_concurrent']}")
        if "max_waiting" in gs:      print(f"    Max queued:       {gs['max_waiting']}")
        print(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    levels = "_".join(str(n) for n in args.n)
    result_path = os.path.join(args.output_dir, f"stress_{levels}_{ts}.json")
    with open(result_path, "w") as f:
        json.dump({
            "timestamp": ts, "server": args.api_url, "model": args.model,
            "concurrency_levels": args.n, "batches": all_results, "gpu_overall": gs,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  RAMP-UP SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Agents':>7} | {'Success':>8} | {'Wall(s)':>8} | {'Avg(s)':>7} | {'Gen tok/s':>9} | {'KV Peak':>8} | {'MaxConc':>7}")
        print(f"  {'─'*7} | {'─'*8} | {'─'*8} | {'─'*7} | {'─'*9} | {'─'*8} | {'─'*7}")
        for b in all_results:
            kv = f"{b['gpu']['kv_cache_pct_max']}%" if b.get("gpu") and "kv_cache_pct_max" in b.get("gpu", {}) else "N/A"
            mc = str(b["gpu"]["max_concurrent"]) if b.get("gpu") and "max_concurrent" in b.get("gpu", {}) else "N/A"
            print(f"  {b['n_agents']:>7} | {b['n_success']}/{b['n_agents']:>5} | {b['total_time_s']:>8} | {b['avg_agent_time_s']:>7} | {b.get('gen_tok_per_s',0):>9.1f} | {kv:>8} | {mc:>7}")
        print(f"  {'─'*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
