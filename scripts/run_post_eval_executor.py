from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch

try:
    # Local absolute-style imports relative to repo root
    from src.config import GlobalPathConfig, DEFAULT_START_TASK_NAME
    from src.plan import Plan
    from src.data_loader import TaskDataset
    from src.metrics.evaluator import calculate_task_score, calculate_qop
except Exception:
    # Add catp_base to sys.path if needed
    import sys as _sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    catp_base_dir = repo_root / "catp_base"
    if str(catp_base_dir) not in _sys.path:
        _sys.path.insert(0, str(catp_base_dir))
    from src.config import GlobalPathConfig, DEFAULT_START_TASK_NAME  # type: ignore
    from src.plan import Plan  # type: ignore
    from src.data_loader import TaskDataset  # type: ignore
    from src.metrics.evaluator import calculate_task_score, calculate_qop  # type: ignore


def _resolve_prev_deps(plan: List[Any]) -> List[Any]:
    """Resolve 'output_of_previous_tool' into concrete tool names.

    - For each [tool, [deps]] pair, replace any 'output_of_previous_tool' with the
      immediately previous tool name. If none exists (first tool), replace with
      DEFAULT_START_TASK_NAME to keep the plan valid.
    - Returns a new plan list with the same structure.
    """
    out: List[Any] = []
    last_tool: str | None = None
    n = len(plan)
    i = 0
    while i + 1 < n:
        tool = str(plan[i])
        deps = plan[i + 1]
        deps_list = list(deps) if isinstance(deps, list) else [deps]
        new_deps: List[str] = []
        for dep in deps_list:
            d = str(dep)
            if d == "output_of_previous_tool":
                new_deps.append(last_tool if last_tool is not None else DEFAULT_START_TASK_NAME)
            else:
                new_deps.append(d)
        out.append(tool)
        out.append(new_deps)
        last_tool = tool
        i += 2
    return out


def _run_plan(task_id: int, sample_id: int, plan_list: List[Any]) -> Dict[str, Any]:
    """Execute a plan for a specific (task_id, sample_id) and collect metrics.

    Returns a dict with validity, optional error, and metrics when valid.
    """
    # Load sample data directly without iterating all samples
    dataset = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
    if sample_id not in dataset.input_data or sample_id not in dataset.output_data:
        return {
            "valid": False,
            "error": f"Sample {sample_id} not found for task {task_id}",
        }

    input_data = dataset.input_data[sample_id]
    output_data = dataset.output_data[sample_id]

    plan = Plan(plan_list)
    try:
        result = plan.execute(input_data)
    except torch.OutOfMemoryError:
        return {
            "valid": False,
            "oom": True,
            "error": "OutOfMemory during execution",
        }

    if result is None:
        return {
            "valid": False,
            "error": "Invalid plan or execution failure",
        }

    task_score = calculate_task_score(result, output_data, sequential=task_id < 200)
    cost_price = plan.price
    exec_time = plan.exec_time
    qop = calculate_qop(task_score, cost_price)

    return {
        "valid": True,
        "task_score": float(task_score),
        "cost_price": float(cost_price),
        "exec_time": float(exec_time),
        "qop": float(qop),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute gold and predicted plans from plans_post_eval.json and save results.")
    parser.add_argument("--input", default="plans_post_eval.json", help="Path to plans_post_eval.json from run_state_plan_extractor.py")
    parser.add_argument("--output_dir", default="results_post_eval", help="Directory to save the results JSON")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of items to process (0 = no limit)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        sys.stderr.write(f"Input file not found: {args.input}\n")
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not items:
        sys.stderr.write("No items found in input JSON.\n")
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    out_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(items):
        if args.limit and idx >= args.limit:
            break
        try:
            task_id = int(rec["task_id"])  # type: ignore[index]
            sample_id = int(rec["sample_id"])  # type: ignore[index]
            gold_plan = rec.get("gold_plan")
            pred_plan = rec.get("pred_plan")

            # Resolve shorthand dependencies for execution
            gold_exec = _resolve_prev_deps(gold_plan) if isinstance(gold_plan, list) else None
            pred_exec = _resolve_prev_deps(pred_plan) if isinstance(pred_plan, list) else None

            if gold_exec is None or pred_exec is None:
                out_records.append({
                    "task_id": task_id,
                    "sample_id": sample_id,
                    "error": "Missing or malformed plan(s)",
                })
                continue

            gold_res = _run_plan(task_id, sample_id, gold_exec)
            pred_res = _run_plan(task_id, sample_id, pred_exec)

            out_records.append({
                "task_id": task_id,
                "sample_id": sample_id,
                "gold": gold_res,
                "pred": pred_res,
            })
        except Exception as e:
            out_records.append({
                "task_id": rec.get("task_id"),
                "sample_id": rec.get("sample_id"),
                "error": f"Exception: {e}",
            })

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f"results_{base}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(out_records), "items": out_records}, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

