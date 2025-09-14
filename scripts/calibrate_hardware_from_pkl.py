import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import psutil
import torch

try:
    from src.config import GlobalMetricsConfig, GlobalPathConfig
    from src.plan import Plan
    from src.data_loader import TaskDataset
    from src.tools import tool_manager
    from src.utils import normalize_task_name
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
    from src.tools import tool_manager  # type: ignore
    from src.utils import normalize_task_name  # type: ignore


def _resolve_prev_deps(plan: List[Any], default_start: str) -> List[Any]:
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
                new_deps.append(last_tool if last_tool is not None else default_start)
            else:
                new_deps.append(d)
        out.append(tool)
        out.append(new_deps)
        last_tool = tool
        i += 2
    return out


def _gather_tools_from_pkls(pkl_paths: List[str]) -> Set[str]:
    tools: Set[str] = set()
    for p in pkl_paths:
        obj = pickle.load(open(p, "rb"))
        plan_pools = obj if isinstance(obj, list) else [obj]
        for pool in plan_pools:
            for task_id, samples in pool.plans.items():
                for sample_id in samples.keys():
                    plans = pool.plans[task_id][sample_id]
                    plans = [plans] if isinstance(plans, tuple) else plans
                    for pl in plans:
                        for i in range(0, len(pl), 2):
                            t = str(pl[i])
                            if t == "output_of_previous_tool":
                                continue
                            tools.add(normalize_task_name(t))
    return tools


def _measure_model_param_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def _measure_tool_memory_mb(tool_name: str, device: str) -> Tuple[float, float]:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tool = tool_manager.get_model(tool_name, None)
    cpu_bytes = _measure_model_param_bytes(tool.model)
    cpu_mb = cpu_bytes / (1024 ** 2)
    if device.startswith("cuda") and torch.cuda.is_available():
        dev_index = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.synchronize(dev_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev_index)
        base_alloc = torch.cuda.memory_allocated(dev_index)
        tool.to(device)
        torch.cuda.synchronize(dev_index)
        alloc = torch.cuda.memory_allocated(dev_index)
        gpu_mb = max(0.0, (alloc - base_alloc) / (1024 ** 2))
        tool.to("cpu")
        torch.cuda.empty_cache()
    else:
        gpu_mb = 0.0
    _ = proc.memory_info().rss - rss_before
    return float(cpu_mb), float(gpu_mb)


def _update_global_long_term_maps(cpu_map: Dict[str, float], gpu_map: Dict[str, float]) -> None:
    GlobalMetricsConfig.tools_cpu_long_term_mem.update(cpu_map)
    GlobalMetricsConfig.tools_gpu_long_term_mem.update(gpu_map)


def _run_plan_for_cost(task_id: int, sample_id: int, plan_list: List[Any]) -> float | None:
    dataset = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
    if sample_id not in dataset.input_data or sample_id not in dataset.output_data:
        return None
    input_data = dataset.input_data[sample_id]
    plan = Plan(plan_list)
    try:
        _ = plan.execute(input_data)
    except torch.OutOfMemoryError:
        return None
    except Exception:
        return None
    return float(plan.price)


def _compute_cost_bounds(pkl_paths: List[str], device: str, limit: int) -> Tuple[float, float, int]:
    seen = 0
    min_cost = float("inf")
    max_cost = float("-inf")
    default_start = "input_of_query"
    for p in pkl_paths:
        obj = pickle.load(open(p, "rb"))
        plan_pools = obj if isinstance(obj, list) else [obj]
        for pool in plan_pools:
            for task_id, samples in pool.plans.items():
                for sample_id in samples.keys():
                    plans = pool.plans[task_id][sample_id]
                    plans = [plans] if isinstance(plans, tuple) else plans
                    for pl in plans:
                        exec_plan = _resolve_prev_deps(pl, default_start)
                        c = _run_plan_for_cost(int(task_id), int(sample_id), exec_plan)
                        if c is None:
                            continue
                        min_cost = min(min_cost, c)
                        max_cost = max(max_cost, c)
                        seen += 1
                        if limit and seen >= limit:
                            return (min_cost if min_cost != float("inf") else 0.0,
                                    max_cost if max_cost != float("-inf") else 0.0,
                                    seen)
    return (min_cost if min_cost != float("inf") else 0.0,
            max_cost if max_cost != float("-inf") else 0.0,
            seen)


def _detect_hardware() -> Dict[str, Any]:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total_mb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            gpus.append({"id": i, "name": name, "total_memory_mb": int(total_mb)})
    return {
        "gpus": gpus,
        "cpu_count": os.cpu_count(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkls", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plan_limit", type=int, default=50)
    parser.add_argument("--measure_all_registry", action="store_true")
    args = parser.parse_args()

    start = time.time()

    tools = _gather_tools_from_pkls(args.pkls)
    if args.measure_all_registry:
        from src.config import MODEL_REGISTRY
        tools = set(MODEL_REGISTRY.keys())

    cpu_map: Dict[str, float] = {}
    gpu_map: Dict[str, float] = {}
    for t in sorted(tools):
        cpu_mb, gpu_mb = _measure_tool_memory_mb(t, args.device)
        cpu_map[t] = round(cpu_mb, 3)
        gpu_map[t] = round(gpu_mb, 3)

    _update_global_long_term_maps(cpu_map, gpu_map)

    min_cost, max_cost, seen = _compute_cost_bounds(args.pkls, args.device, args.plan_limit)

    out = {
        "hardware": _detect_hardware(),
        "recommended": {
            "tools_cpu_long_term_mem": cpu_map,
            "tools_gpu_long_term_mem": gpu_map,
            "MIN_COST": round(min_cost, 6),
            "MAX_COST": round(max_cost, 6),
        },
        "plans_evaluated": seen,
        "timing_sec": round(time.time() - start, 3),
        "notes": {
            "set_in_config": [
                "GlobalMetricsConfig.tools_cpu_long_term_mem",
                "GlobalMetricsConfig.tools_gpu_long_term_mem",
                "GlobalMetricsConfig.MIN_COST",
                "GlobalMetricsConfig.MAX_COST",
            ]
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"written": args.out, "plans_evaluated": seen}, indent=2))


if __name__ == "__main__":
    main()


