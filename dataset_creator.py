import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from src.config import GlobalMetricsConfig, GlobalPathConfig
from src.catpllm.utils.cost_utils import calc_plan_price
from src.metrics.evaluator import calculate_qop, calculate_task_score
from src.plan import Plan
from src.data_loader import TaskDataset


_TASK_DESCRIPTIONS: List[str] = []


def _load_task_descriptions(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _task_query_for(task_id: int) -> str | None:
    if 0 <= task_id < len(_TASK_DESCRIPTIONS):
        return _TASK_DESCRIPTIONS[task_id]
    return None


def _ensure_list_shape(plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem):
    if isinstance(scores, tuple):
        return [plans], [scores], [tools_exec_time], [tools_cpu_mem], [tools_gpu_mem]
    return plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem


def _normalize_score(score_tuple: Tuple[float, str]) -> float:
    value, kind = score_tuple
    if kind == "vit_score":
        return value / 100.0
    return value


def _is_valid_score(score_tuple: Tuple[float, str] | None) -> bool:
    if score_tuple is None:
        return False
    value, _ = score_tuple
    return value is not None and value != -2


def _extract_tools(plan: List[Any]) -> List[str]:
    return [plan[i] for i in range(0, len(plan), 2)]


def _compute_price_and_time(plan: List[Any], exec_times: List[float], cpu_short: List[float], gpu_short: List[float]) -> Tuple[float, float]:
    tools = _extract_tools(plan)
    cpu_long = [GlobalMetricsConfig.tools_cpu_long_term_mem[t] for t in tools]
    gpu_long = [GlobalMetricsConfig.tools_gpu_long_term_mem[t] for t in tools]
    price = calc_plan_price(exec_times, cpu_long, cpu_short, gpu_long, gpu_short)
    exec_time_total = float(sum(exec_times))
    return price, exec_time_total


def _recompute_runtime_metrics(
    plan: List[Any],
    task_id: int,
    sample_id: int,
    *,
    data_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Recompute task_score, cost_price, exec_time, and qop by actually executing the plan
    on the specified sample — matching test_opencatp.py logic exactly.

    Returns a dict with keys: task_score, cost_price, exec_time, qop; or None if execution fails.
    """
    try:
        ds = TaskDataset(data_path or GlobalPathConfig.data_path, task_id=task_id)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        # Find the requested sample
        batch = None
        for b in dl:
            if int(b["sample_id"]) == int(sample_id):
                batch = b
                break
        if batch is None:
            return None

        p = Plan(plan)
        try:
            result = p.execute(batch["input"], cost_aware=True)
        except torch.OutOfMemoryError:
            return None

        if result is None:
            return None

        task_score = calculate_task_score(result, batch["output"], sequential=task_id < 200)
        cost_price = float(p.price)
        exec_time = float(p.exec_time)
        qop = float(calculate_qop(task_score, cost_price))

        return {
            "task_score": float(task_score),
            "cost_price": cost_price,
            "exec_time": exec_time,
            "qop": qop,
        }
    except Exception:
        return None


def _stringify_plan(plan: Any) -> str:
    try:
        return str(plan)
    except Exception:
        return json.dumps(plan, ensure_ascii=False)


def process_pkl(
    path: str,
    *,
    recompute_runtime: bool = False,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Dict[int, Any]], Dict[int, Dict[int, Any]], Dict[int, Dict[int, List[Any]]], Dict[int, Dict[int, List[Any]]]]:
    obj = pickle.load(open(path, "rb"))
    plan_pools = obj if isinstance(obj, list) else [obj]

    valid_best: Dict[int, Dict[int, Any]] = {}
    invalid_best: Dict[int, Dict[int, Any]] = {}
    valid_all: Dict[int, Dict[int, List[Any]]] = {}
    invalid_all: Dict[int, Dict[int, List[Any]]] = {}

    for plan_pool in plan_pools:
        for task_id, samples in plan_pool.plans.items():
            for sample_id in samples.keys():
                plans = plan_pool.plans[task_id][sample_id]
                scores = plan_pool.scores[task_id][sample_id]
                tools_exec_time = plan_pool.tools_exec_time[task_id][sample_id]
                tools_cpu_mem = plan_pool.tools_cpu_mem[task_id][sample_id]
                tools_gpu_mem = plan_pool.tools_gpu_mem[task_id][sample_id]

                plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem = _ensure_list_shape(
                    plans, scores, tools_exec_time, tools_cpu_mem, tools_gpu_mem
                )

                valid_variants = []
                invalid_variants = []
                for n in range(len(plans)):
                    plan = plans[n]
                    score_tuple = scores[n]
                    if not _is_valid_score(score_tuple):
                        invalid_variants.append({
                            "plan": _stringify_plan(plan),
                            "task_score": GlobalMetricsConfig.score_penalty,
                            "cost_price": GlobalMetricsConfig.cost_penalty,
                            "exec_time": None,
                            "qop": None,
                            "task_query": _task_query_for(task_id)
                        })
                        continue

                    score_value = _normalize_score(score_tuple)
                    exec_times = list(tools_exec_time[n]) if tools_exec_time[n] is not None else None
                    cpu_short = tools_cpu_mem[n] if tools_cpu_mem[n] is not None else None
                    gpu_short = tools_gpu_mem[n] if tools_gpu_mem[n] is not None else None

                    if exec_times is None or cpu_short is None or gpu_short is None:
                        invalid_variants.append({
                            "plan": _stringify_plan(plan),
                            "task_score": GlobalMetricsConfig.score_penalty,
                            "cost_price": GlobalMetricsConfig.cost_penalty,
                            "exec_time": None,
                            "qop": None,
                            "task_query": _task_query_for(task_id)
                        })
                        continue

                    cost_price, exec_time_total = _compute_price_and_time(plan, exec_times, cpu_short, gpu_short)
                    qop = calculate_qop(score_value, cost_price)

                    variant = {
                        "plan": _stringify_plan(plan),
                        "task_score": float(score_value),
                        "cost_price": float(cost_price),
                        "exec_time": float(exec_time_total),
                        "qop": float(qop),
                        "task_query": _task_query_for(task_id)
                    }

                    # Optional: recompute using the exact runtime method used in testing.
                    if recompute_runtime:
                        recomputed = _recompute_runtime_metrics(plan, task_id, sample_id, data_path=data_path)
                        if recomputed is not None:
                            variant.update(recomputed)

                    valid_variants.append(variant)

                if valid_variants:
                    best = max(valid_variants, key=lambda x: x["qop"])
                    valid_best.setdefault(task_id, {})[sample_id] = best
                    valid_all.setdefault(task_id, {})[sample_id] = valid_variants
                if invalid_variants and not valid_variants:
                    worst = invalid_variants[0]
                    invalid_best.setdefault(task_id, {})[sample_id] = worst
                    invalid_all.setdefault(task_id, {})[sample_id] = invalid_variants

    return valid_best, invalid_best, valid_all, invalid_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkls", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--recompute_runtime", action="store_true", help="Recompute score/price per plan by executing like test_opencatp.py")
    parser.add_argument("--data_path", type=str, default=GlobalPathConfig.data_path)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global _TASK_DESCRIPTIONS
    repo_root = Path(__file__).parent
    _TASK_DESCRIPTIONS = _load_task_descriptions(repo_root / "catp_base/dataset/task_descriptions.txt")

    merged_valid_best: Dict[int, Dict[int, Any]] = {}
    merged_invalid_best: Dict[int, Dict[int, Any]] = {}
    merged_valid_all: Dict[int, Dict[int, List[Any]]] = {}
    merged_invalid_all: Dict[int, Dict[int, List[Any]]] = {}

    for pkl_path in args.pkls:
        v_best, i_best, v_all, i_all = process_pkl(
            pkl_path,
            recompute_runtime=args.recompute_runtime,
            data_path=args.data_path,
        )
        for tid, samples in v_best.items():
            merged_valid_best.setdefault(tid, {}).update(samples)
        for tid, samples in i_best.items():
            merged_invalid_best.setdefault(tid, {}).update(samples)
        for tid, samples in v_all.items():
            merged_valid_all.setdefault(tid, {}).update(samples)
        for tid, samples in i_all.items():
            merged_invalid_all.setdefault(tid, {}).update(samples)

    with open(os.path.join(args.out_dir, "valid_plans_best.json"), "w", encoding="utf-8") as f:
        json.dump(merged_valid_best, f, indent=2)
    with open(os.path.join(args.out_dir, "invalid_plans_best.json"), "w", encoding="utf-8") as f:
        json.dump(merged_invalid_best, f, indent=2)
    with open(os.path.join(args.out_dir, "valid_plans_all.json"), "w", encoding="utf-8") as f:
        json.dump(merged_valid_all, f, indent=2)
    with open(os.path.join(args.out_dir, "invalid_plans_all.json"), "w", encoding="utf-8") as f:
        json.dump(merged_invalid_all, f, indent=2)


if __name__ == "__main__":
    main()
