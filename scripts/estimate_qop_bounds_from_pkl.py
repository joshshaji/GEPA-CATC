import argparse
import ast
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from src.config import GlobalMetricsConfig, GlobalPathConfig
    from src.plan import Plan
    from src.data_loader import TaskDataset
    from src.metrics.evaluator import calculate_task_score
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    catp_base_dir = repo_root / "catp_base"
    if str(catp_base_dir) not in _sys.path:
        _sys.path.insert(0, str(catp_base_dir))
    from src.config import GlobalMetricsConfig, GlobalPathConfig  # type: ignore
    from src.plan import Plan  # type: ignore
    from src.data_loader import TaskDataset  # type: ignore
    from src.metrics.evaluator import calculate_task_score  # type: ignore


def _ensure_list_shape(plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem):
    if isinstance(scores, tuple):
        return [plans], [scores], [tools_exec_time], [tools_cpu_mem], [tools_gpu_mem]
    return plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem


def _parse_plan(plan_any: Any) -> List[Any]:
    if isinstance(plan_any, list):
        return plan_any
    if isinstance(plan_any, str):
        return ast.literal_eval(plan_any)
    raise ValueError("Unsupported plan representation")


def _resolve_prev_deps(plan: List[Any]) -> List[Any]:
    out: List[Any] = []
    last_tool: str | None = None
    i = 0
    n = len(plan)
    while i + 1 < n:
        tool = str(plan[i])
        deps = plan[i + 1]
        deps_list = list(deps) if isinstance(deps, list) else [deps]
        new_deps: List[str] = []
        for dep in deps_list:
            d = str(dep)
            if d == "output_of_previous_tool":
                new_deps.append(last_tool if last_tool is not None else "input_of_query")
            else:
                new_deps.append(d)
        out.append(tool)
        out.append(new_deps)
        last_tool = tool
        i += 2
    return out


@torch.inference_mode()
def _run_one(task_id: int, sample_id: int, plan_list: List[Any]) -> Tuple[float, float] | None:
    dataset = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
    if sample_id not in dataset.input_data or sample_id not in dataset.output_data:
        return None
    input_data = dataset.input_data[sample_id]
    output_data = dataset.output_data[sample_id]
    plan = Plan(plan_list)
    try:
        result = plan.execute(input_data)
    except torch.OutOfMemoryError:
        return None
    if result is None:
        return None
    task_score = float(calculate_task_score(result, output_data, sequential=task_id < 200))
    cost_price = float(plan.price)
    return task_score, cost_price


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkls", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--per-sample-variants", type=int, default=1)
    parser.add_argument("--quantile-low", type=float, default=0.01)
    parser.add_argument("--quantile-high", type=float, default=0.99)
    args = parser.parse_args()

    pairs: List[Tuple[float, float]] = []
    picked = 0

    for pkl_path in args.pkls:
        obj = pickle.load(open(pkl_path, "rb"))
        plan_pools = obj if isinstance(obj, list) else [obj]
        for plan_pool in plan_pools:
            for task_id, samples in plan_pool.plans.items():
                for sample_id in samples.keys():
                    plans = plan_pool.plans[task_id][sample_id]
                    scores = plan_pool.scores[task_id][sample_id]
                    tools_exec_time = plan_pool.tools_exec_time[task_id][sample_id]
                    tools_cpu_mem = plan_pool.tools_cpu_mem[task_id][sample_id]
                    tools_gpu_mem = plan_pool.tools_gpu_mem[task_id][sample_id]
                    plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem = _ensure_list_shape(plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem)
                    max_variants = min(args.per_sample_variants, len(plans))
                    for n in range(max_variants):
                        raw_plan = plans[n]
                        try:
                            plan_list = _parse_plan(raw_plan)
                            exec_plan = _resolve_prev_deps(plan_list)
                        except Exception:
                            continue
                        res = _run_one(int(task_id), int(sample_id), exec_plan)
                        if res is None:
                            continue
                        pairs.append(res)
                        picked += 1
                        if args.sample_limit and picked >= args.sample_limit:
                            break
                    if args.sample_limit and picked >= args.sample_limit:
                        break
                if args.sample_limit and picked >= args.sample_limit:
                    break
            if args.sample_limit and picked >= args.sample_limit:
                break

    scores = [s for s, _ in pairs]
    costs = [c for _, c in pairs]
    if scores:
        min_score_obs = float(np.min(scores))
        max_score_obs = float(np.max(scores))
    else:
        min_score_obs = GlobalMetricsConfig.MIN_SCORE
        max_score_obs = GlobalMetricsConfig.MAX_SCORE
    if costs:
        lo = max(0, int(len(costs) * args.quantile_low) - 1)
        hi = min(len(costs) - 1, int(len(costs) * args.quantile_high))
        costs_sorted = sorted(costs)
        min_cost_suggest = float(costs_sorted[lo])
        max_cost_suggest = float(costs_sorted[hi])
    else:
        min_cost_suggest = GlobalMetricsConfig.MIN_COST
        max_cost_suggest = GlobalMetricsConfig.MAX_COST

    report = {
        "num_executed": len(pairs),
        "score_bounds_observed": {
            "min": min_score_obs,
            "max": max_score_obs,
        },
        "cost_bounds_suggested": {
            "min": min_cost_suggest,
            "max": max_cost_suggest,
        },
        "config_snippet": {
            "MIN_SCORE": 0.0,
            "MAX_SCORE": 1.0,
            "MIN_COST": min_cost_suggest,
            "MAX_COST": max_cost_suggest,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


