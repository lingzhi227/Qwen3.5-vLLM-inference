#!/bin/bash
# =============================================================================
# BixBench Test: Solve bix-1-q1 using Qwen3.5-9B on local PCIE A100
# Usage: bash scripts/PCIE/test_bixbench.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
SCRIPT_DIR="$WORKDIR/scripts/PCIE"
NODE_INFO="$WORKDIR/.node_info_pcie"
MAMBA_BIN="/pscratch/sd/l/lingzhi/bin/micromamba"
export MAMBA_ROOT_PREFIX="/pscratch/sd/l/lingzhi/micromamba"

# Get API URL
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi
API_URL="${API_URL:-http://localhost:8000/v1}"

echo "=========================================="
echo "  BixBench bix-1-q1 — Qwen3.5-9B (PCIE)"
echo "  vLLM: $API_URL"
echo "=========================================="

# Health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: vLLM not responding at $API_URL"
    echo "Start it first: bash scripts/PCIE/serve.sh"
    exit 1
fi

MODEL_NAME=$(curl -s "${API_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: $MODEL_NAME"
echo ""

# Run in bixbench conda env (has R, rpy2, jupyter, etc.)
export BIXBENCH_API_URL="$API_URL"
export BIXBENCH_MODEL="$MODEL_NAME"
export BIXBENCH_SCRIPT_DIR="$SCRIPT_DIR"

"$MAMBA_BIN" run -n bixbench python3 - << 'PYEOF'
import asyncio
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
BASE_URL = os.environ["BIXBENCH_API_URL"]
MODEL = os.environ["BIXBENCH_MODEL"]
SCRIPT_DIR = Path(os.environ["BIXBENCH_SCRIPT_DIR"])
WORK_DIR = SCRIPT_DIR / "bixbench_workspace" / "bix-1-q1"
DATA_DIR = Path("/pscratch/sd/l/lingzhi/Projects/BixBench/bixbench_data")
NB_PATH = WORK_DIR / "notebook.ipynb"
MAX_ROUNDS = 40
NB_OUTPUT_LIMIT = 2000
MAX_TOKENS = 8192

print(f"[Config] vLLM: {BASE_URL}")
print(f"[Config] Model: {MODEL}")
print(f"[Config] Workspace: {WORK_DIR}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_capsule_data():
    capsule_zip = "CapsuleFolder-33b801bb-9b47-4a0a-9314-05325c82fde7.zip"
    zip_path = DATA_DIR / capsule_zip

    if not zip_path.exists():
        print("[Data] Downloading capsule from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="futurehouse/BixBench",
            filename=capsule_zip,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
    else:
        print(f"[Data] Using cached capsule: {zip_path}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("[Data] Extracting capsule data...")
    with zipfile.ZipFile(zip_path) as zf:
        data_prefix = "CapsuleData-33b801bb-9b47-4a0a-9314-05325c82fde7/"
        for member in zf.namelist():
            if member.startswith(data_prefix) and not member.endswith("/"):
                filename = member[len(data_prefix):]
                target = WORK_DIR / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, str(NB_PATH))

    files = [f for f in WORK_DIR.iterdir() if f.name != "notebook.ipynb"]
    print(f"[Data] Extracted {len(files)} files:")
    for f in sorted(files):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Jupyter Kernel Manager
# ---------------------------------------------------------------------------
class NotebookRunner:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.km = None
        self.kc = None
        self.nb = nbformat.v4.new_notebook()
        self.nb_path = work_dir / "notebook.ipynb"

    async def start(self):
        self.km = AsyncKernelManager(kernel_name="python3")
        await self.km.start_kernel(cwd=str(self.work_dir))
        self.kc = self.km.client()
        self.kc.start_channels()
        await self.kc.wait_for_ready(timeout=30)
        print("[Kernel] Python kernel started")
        try:
            await self._execute_silent("%load_ext rpy2.ipython")
            print("[Kernel] rpy2 R magic loaded")
        except Exception:
            print("[Kernel] rpy2 not available")

    async def stop(self):
        if self.kc:
            self.kc.stop_channels()
        if self.km:
            await self.km.shutdown_kernel(now=True)
        print("[Kernel] Stopped")

    async def _execute_silent(self, code):
        msg_id = self.kc.execute(code, silent=True)
        while True:
            msg = await self.kc.get_iopub_msg(timeout=30)
            if msg["parent_header"].get("msg_id") == msg_id:
                if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                    break

    async def execute_cell(self, code, cell_idx=None):
        cell = nbformat.v4.new_code_cell(source=code)
        if cell_idx is None or cell_idx >= len(self.nb.cells):
            self.nb.cells.append(cell)
            cell_idx = len(self.nb.cells) - 1
        else:
            self.nb.cells[cell_idx] = cell

        msg_id = self.kc.execute(code)
        outputs = []
        cell_outputs = []
        error_output = None

        while True:
            try:
                msg = await asyncio.wait_for(self.kc.get_iopub_msg(), timeout=600)
            except asyncio.TimeoutError:
                outputs.append("[TIMEOUT: 600s]")
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                text = content.get("text", "")
                outputs.append(text)
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="stream", name=content.get("name", "stdout"), text=text))

            elif msg_type == "execute_result":
                text = content.get("data", {}).get("text/plain", "")
                outputs.append(text)
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="execute_result",
                    data=content.get("data", {}),
                    metadata=content.get("metadata", {}),
                    execution_count=content.get("execution_count")))

            elif msg_type == "display_data":
                data = content.get("data", {})
                if "text/plain" in data:
                    outputs.append(data["text/plain"])
                if "image/png" in data:
                    outputs.append("[Image]")
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="display_data", data=data,
                    metadata=content.get("metadata", {})))

            elif msg_type == "error":
                ename = content.get("ename", "Error")
                evalue = content.get("evalue", "")
                tb = content.get("traceback", [])
                clean_tb = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in tb]
                error_text = f"{ename}: {evalue}\n" + "\n".join(clean_tb[-5:])
                outputs.append(error_text)
                error_output = error_text
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="error", ename=ename, evalue=evalue,
                    traceback=content.get("traceback", [])))

            elif msg_type == "status":
                if content["execution_state"] == "idle":
                    break

        self.nb.cells[cell_idx].outputs = cell_outputs
        nbformat.write(self.nb, str(self.nb_path))

        output_text = "".join(outputs)
        if len(output_text) > NB_OUTPUT_LIMIT:
            half = NB_OUTPUT_LIMIT // 2
            output_text = (output_text[:half]
                           + f"\n...[truncated {len(output_text)} chars]...\n"
                           + output_text[-half:])

        return output_text, error_output is not None

    def list_workdir(self):
        entries = []
        for entry in sorted(self.work_dir.iterdir()):
            if entry.name == "notebook.ipynb":
                continue
            info = {"name": entry.name, "type": "dir" if entry.is_dir() else "file"}
            if entry.is_file():
                info["size"] = entry.stat().st_size
            entries.append(info)
        return json.dumps({"files": entries}, indent=2)

    def render_notebook(self):
        if not self.nb.cells:
            return "Notebook empty. Use edit_cell to add cells."

        parts = []
        for i, cell in enumerate(self.nb.cells):
            parts.append(f"### Cell [{i}]")
            parts.append(f"```python\n{cell.source}\n```")

            if cell.outputs:
                out_parts = []
                for out in cell.outputs:
                    if out.output_type == "stream":
                        out_parts.append(out.text)
                    elif out.output_type == "execute_result":
                        out_parts.append(out.data.get("text/plain", ""))
                    elif out.output_type == "error":
                        out_parts.append(f"{out.ename}: {out.evalue}")
                    elif out.output_type == "display_data":
                        if "text/plain" in out.data:
                            out_parts.append(out.data["text/plain"])
                        if "image/png" in out.data:
                            out_parts.append("[Image]")

                output = "".join(out_parts)
                if len(output) > NB_OUTPUT_LIMIT:
                    output = output[:NB_OUTPUT_LIMIT] + "\n...[truncated]"
                parts.append(f"**Output:**\n```\n{output}\n```")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_cell",
            "description": (
                "Create or edit a code cell in the Jupyter notebook. "
                "The cell is automatically executed and the output is returned. "
                "If idx is not provided, a new cell is appended. "
                "If idx is provided, that cell is replaced. "
                "Variables persist between cells (same kernel). "
                "Use %%R on the first line for R code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contents": {
                        "type": "string",
                        "description": "Python or R code to execute",
                    },
                    "idx": {
                        "type": "integer",
                        "description": "Cell index to replace (optional)",
                    },
                },
                "required": ["contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workdir",
            "description": "List workspace files with sizes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final answer (A/B/C/D). Ends session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Letter A/B/C/D",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Streaming chat — strip thinking from history to save context
# ---------------------------------------------------------------------------
def stream_chat(client, messages, round_num):
    print(f"\n{'='*60}")
    print(f"  Round {round_num}")
    print(f"{'='*60}")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    raw_content_parts = []
    thinking_parts = []
    answer_parts = []
    tool_calls_acc = {}
    finish_reason = None
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
                    before, after = text.split("</think>", 1)
                    if before:
                        if before.strip() and not thinking_header_shown:
                            print(f"\033[36m  [THINKING]\033[0m")
                            thinking_header_shown = True
                        thinking_parts.append(before)
                        if thinking_header_shown:
                            sys.stdout.write(f"\033[90m{before}\033[0m")
                            sys.stdout.flush()
                    state = "answering"
                    after = after.lstrip("\n")
                    if after.strip():
                        if not answer_header_shown:
                            print(f"\n\033[32m  [RESPONSE]\033[0m")
                            answer_header_shown = True
                        answer_parts.append(after)
                        sys.stdout.write(after)
                        sys.stdout.flush()
                else:
                    if text.strip() and not thinking_header_shown:
                        print(f"\033[36m  [THINKING]\033[0m")
                        thinking_header_shown = True
                    thinking_parts.append(text)
                    if thinking_header_shown:
                        sys.stdout.write(f"\033[90m{text}\033[0m")
                        sys.stdout.flush()
            elif state == "answering":
                if text.strip() and not answer_header_shown:
                    print(f"\n\033[32m  [RESPONSE]\033[0m")
                    answer_header_shown = True
                answer_parts.append(text)
                if answer_header_shown:
                    sys.stdout.write(text)
                    sys.stdout.flush()

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

    full_answer = "".join(answer_parts)
    full_content = "".join(raw_content_parts)

    # Strip thinking from history to save context tokens
    content_for_history = full_answer if full_answer else full_content
    content_for_history = re.sub(
        r"<think>.*?</think>", "", content_for_history, flags=re.DOTALL
    ).strip()

    assistant_msg = {"role": "assistant", "content": content_for_history}

    tool_calls_list = []
    if tool_calls_acc:
        tool_calls_list = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in tool_calls_acc.values()
        ]
        assistant_msg["tool_calls"] = tool_calls_list
        if not content_for_history:
            assistant_msg["content"] = None

    messages.append(assistant_msg)
    print()

    return tool_calls_list, full_answer


# ---------------------------------------------------------------------------
# Question & prompts
# ---------------------------------------------------------------------------
QUESTION = (
    'Using the provided RNA-seq count data and metadata files, perform DESeq2 '
    'differential expression analysis to identify significant DEGs (padj < 0.05), '
    'then run enrichGO analysis with clusterProfiler::simplify() (similarity > 0.7). '
    'What is the approximate adjusted p-value (rounded to 4 decimal points) for '
    '"regulation of T cell activation" in the resulting simplified GO enrichment results?'
)
CHOICES = "(A) 0.0002\n(B) 7.820659E-05\n(C) 0.0003\n(D) 1.847038E-05"
IDEAL = "A"


def build_system_prompt():
    return f"""You are an expert bioinformatician working in a Jupyter notebook environment.

You have 3 tools:
1. edit_cell(contents, idx=None) - Write and execute a code cell. Variables persist between cells.
2. list_workdir() - List files in the workspace.
3. submit_answer(answer) - Submit your final answer (a letter A/B/C/D).

ENVIRONMENT:
- Python 3.11 with pandas, numpy, scipy, openpyxl
- R 4.3 via rpy2: use %%R magic on the first line of a cell for R code
- R packages: DESeq2, clusterProfiler, org.Hs.eg.db, readxl
- Use %%R -i var_name to pass Python variables to R
- Use %%R -o var_name to pass R variables to Python

APPROACH:
1. List workspace files to see what data is available
2. Load and inspect data using Python (pandas)
3. For DESeq2 and enrichGO, use R via %%R magic cells
4. Key steps:
   - Read count data and metadata
   - Run DESeq2 differential expression
   - Get significant DEGs (padj < 0.05)
   - Run enrichGO with clusterProfiler
   - Apply simplify(cutoff=0.7)
   - Find the target GO term's adjusted p-value
5. Submit answer when done

CELL EDITING RULES:
- Ensure each cell executes successfully before moving to the next.
- Edit existing cells by their index number when fixing bugs, rather than creating new ones.
  For example, if cell [3] errors, call edit_cell(contents="fixed code...", idx=3) to replace it.
- Only append a new cell (idx=None) when adding genuinely new analysis steps.

TIPS:
- The count data file uses whitespace separators
- The metadata is in xlsx format (use readxl::read_excel in R)
- Gene IDs have version numbers (ENSG00000223972.5) - strip versions for ID mapping
- Use bitr() to convert ENSEMBL to ENTREZID
- The metadata has a 'condition' column (ASXL1 vs Control) and 'sex' column
- Include sex in the DESeq2 design: ~ sex + condition
- R variables persist between %%R cells (same R session via rpy2). Do NOT re-run entire analyses unnecessarily.
- IMPORTANT: Always wrap R library() calls with suppressMessages() to avoid verbose loading messages consuming output, e.g.: suppressMessages({{library(DESeq2); library(clusterProfiler)}})
- IMPORTANT: When calling enrichGO(), use pvalueCutoff=0.05 and qvalueCutoff=0.2 (NOT 1.0). simplify() computes O(n^2) pairwise similarity — passing thousands of terms makes it extremely slow.

Workspace: {WORK_DIR}
"""


def build_user_prompt(notebook_state):
    return f"""{QUESTION}

Answer choices:
{CHOICES}

Current notebook state:
{notebook_state}

Analyze the data step by step using notebook cells. When done, use submit_answer with the letter of your answer."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print(f"\n{'='*60}")
    print(f"  BixBench Test — bix-1-q1")
    print(f"  Model: {MODEL} (PCIE A100)")
    print(f"{'='*60}\n")

    prepare_capsule_data()

    client = OpenAI(base_url=BASE_URL, api_key="empty")
    try:
        models = client.models.list()
        print(f"\n[OK] Connected to vLLM: {models.data[0].id}")
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to vLLM at {BASE_URL}: {e}")
        return 1

    runner = NotebookRunner(WORK_DIR)
    await runner.start()

    start_time = time.time()
    trace_events = []

    try:
        system_prompt = build_system_prompt()
        nb_state = runner.render_notebook()
        user_prompt = build_user_prompt(nb_state)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        submitted_answer = None
        last_nb_tool_idx = None

        for round_num in range(1, MAX_ROUNDS + 1):
            # Hide old notebook views to prevent context bloat
            if last_nb_tool_idx is not None:
                old_msg = messages[last_nb_tool_idx]
                old_content = old_msg.get("content", "")
                marker = "\n\nCurrent notebook:\n"
                if marker in old_content:
                    messages[last_nb_tool_idx] = {
                        **old_msg,
                        "content": old_content[:old_content.index(marker)],
                    }

            tool_calls, answer_text = stream_chat(client, messages, round_num)

            if not tool_calls:
                match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_text, re.I)
                if match:
                    submitted_answer = match.group(1).strip()
                    print(f"\n\033[35m  >>> ANSWER (from text): {submitted_answer}\033[0m")
                letter_match = re.search(r"\b([ABCD])\b", answer_text)
                if not submitted_answer and letter_match:
                    submitted_answer = letter_match.group(1)
                    print(f"\n\033[35m  >>> ANSWER (inferred): {submitted_answer}\033[0m")
                print("[INFO] No tool calls, ending.")
                break

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": f"JSON error: {e}",
                    })
                    continue

                if fn_name == "edit_cell":
                    code = fn_args.get("contents", "")
                    idx = fn_args.get("idx")

                    disp_code = code[:200] + "..." if len(code) > 200 else code
                    print(f"\033[33m  [edit_cell] idx={idx}\033[0m")
                    print(f"\033[90m  {disp_code}\033[0m")

                    output, had_error = await runner.execute_cell(code, idx)

                    print(f"\033[34m  [output] ({len(output)} chars)"
                          f"{' ERROR' if had_error else ''}\033[0m")
                    if output:
                        print(f"\033[90m  {output[:500]}\033[0m")

                    actual_idx = idx if idx is not None else len(runner.nb.cells) - 1
                    nb_view = runner.render_notebook()

                    if had_error:
                        result = (
                            f"Cell [{actual_idx}] executed with ERROR.\n"
                            f"Output:\n{output}\n\n"
                            f"FIX: Call edit_cell(contents=\"...\", idx={actual_idx}) "
                            f"to fix this cell. Do NOT create a new cell.\n\n"
                            f"Current notebook:\n{nb_view}"
                        )
                    else:
                        result = (
                            f"Cell [{actual_idx}] executed.\n"
                            f"Output:\n{output}\n\n"
                            f"Current notebook:\n{nb_view}"
                        )
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": result,
                    })
                    last_nb_tool_idx = len(messages) - 1

                    trace_events.append({
                        "round": round_num, "tool": "edit_cell",
                        "code": code[:500], "output": output[:500],
                    })

                elif fn_name == "list_workdir":
                    print(f"\033[33m  [list_workdir]\033[0m")
                    result = runner.list_workdir()
                    print(f"\033[90m  {result[:300]}\033[0m")
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": result,
                    })

                elif fn_name == "submit_answer":
                    submitted_answer = fn_args.get("answer", "")
                    print(f"\n\033[35m  >>> SUBMITTED ANSWER: {submitted_answer}\033[0m")
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": f"Answer '{submitted_answer}' submitted. Session ended.",
                    })

                else:
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": f"Unknown tool: {fn_name}",
                    })

            if submitted_answer is not None:
                break

    finally:
        await runner.stop()

    # Results
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    final = submitted_answer or "No answer"
    correct = final.upper().strip() in ("A", "0.0002")

    print(f"  Model:         {MODEL}")
    print(f"  Model answer:  {final}")
    print(f"  Ideal answer:  A (0.0002)")
    print(f"  Rounds used:   {round_num}/{MAX_ROUNDS}")
    print(f"  Duration:      {duration:.1f}s")
    print(f"  Grade:         {'CORRECT' if correct else 'INCORRECT'}")
    print(f"{'='*60}")

    # Save trace
    trace_path = str(SCRIPT_DIR / "bixbench_trace_9b_bix1_q1.json")
    with open(trace_path, "w") as f:
        json.dump({
            "model": MODEL,
            "question": "bix-1-q1",
            "answer": final,
            "correct": correct,
            "duration_s": duration,
            "rounds": round_num,
            "events": trace_events,
        }, f, indent=2)
    print(f"  Trace: {trace_path}")

    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
PYEOF
