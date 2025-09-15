import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    from src.config import GlobalMetricsConfig, GlobalPathConfig, DEFAULT_START_TASK_NAME
    from src.metrics.evaluator import calculate_qop, calculate_task_score
    from src.plan import Plan
    from src.data_loader import TaskDataset
    from src.checkpoint import JsonCheckpointer
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    catp_base_dir = repo_root / "catp_base"
    if str(catp_base_dir) not in _sys.path:
        _sys.path.insert(0, str(catp_base_dir))
    from src.config import GlobalMetricsConfig, GlobalPathConfig, DEFAULT_START_TASK_NAME  # type: ignore
    from src.metrics.evaluator import calculate_qop, calculate_task_score  # type: ignore
    from src.plan import Plan  # type: ignore
    from src.data_loader import TaskDataset  # type: ignore
    from src.checkpoint import JsonCheckpointer  # type: ignore


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


def _stringify_plan(plan: Any) -> str:
    try:
        return str(plan)
    except Exception:
        return json.dumps(plan, ensure_ascii=False)


def _resolve_prev_deps(plan: List[Any]) -> List[Any]:
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
    dataset = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
    if sample_id not in dataset.input_data or sample_id not in dataset.output_data:
        return {"valid": False, "error": f"Sample {sample_id} not found for task {task_id}"}

    input_data = dataset.input_data[sample_id]
    output_data = dataset.output_data[sample_id]

    plan = Plan(plan_list)
    try:
        result = plan.execute(input_data)
    except torch.OutOfMemoryError:
        return {"valid": False, "oom": True, "error": "OutOfMemory during execution"}
    except Exception as e:
        return {"valid": False, "error": f"Exception during execution: {e}"}

    if result is None:
        return {"valid": False, "error": "Invalid plan or execution failure"}

    task_score = float(calculate_task_score(result, output_data, sequential=task_id < 200))
    cost_price = float(plan.price)
    exec_time = float(plan.exec_time)
    qop = float(calculate_qop(task_score, cost_price))

    return {
        "valid": True,
        "task_score": task_score,
        "cost_price": cost_price,
        "exec_time": exec_time,
        "qop": qop,
    }


def process_pkl_execute(path: str) -> Tuple[Dict[int, Dict[int, Any]], Dict[int, Dict[int, Any]], Dict[int, Dict[int, List[Any]]], Dict[int, Dict[int, List[Any]]]]:
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
                if isinstance(plans, tuple):
                    plans = [plans]

                valid_variants = []
                invalid_variants = []
                for plan in plans:
                    plan_exec = _resolve_prev_deps(plan)
                    res = _run_plan(int(task_id), int(sample_id), plan_exec)
                    if not res.get("valid"):
                        invalid_variants.append({
                            "plan": _stringify_plan(plan),
                            "task_score": GlobalMetricsConfig.score_penalty,
                            "cost_price": GlobalMetricsConfig.cost_penalty,
                            "exec_time": None,
                            "qop": None,
                            "task_query": _task_query_for(int(task_id)),
                        })
                    else:
                        valid_variants.append({
                            "plan": _stringify_plan(plan),
                            "task_score": res["task_score"],
                            "cost_price": res["cost_price"],
                            "exec_time": res["exec_time"],
                            "qop": res["qop"],
                            "task_query": _task_query_for(int(task_id)),
                        })

                if valid_variants:
                    best = max(valid_variants, key=lambda x: x["qop"])
                    valid_best.setdefault(int(task_id), {})[int(sample_id)] = best
                    valid_all.setdefault(int(task_id), {})[int(sample_id)] = valid_variants
                if invalid_variants and not valid_variants:
                    worst = invalid_variants[0]
                    invalid_best.setdefault(int(task_id), {})[int(sample_id)] = worst
                    invalid_all.setdefault(int(task_id), {})[int(sample_id)] = invalid_variants

    return valid_best, invalid_best, valid_all, invalid_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkls", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task_descriptions", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--autosave_steps", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global _TASK_DESCRIPTIONS
    repo_root = Path(__file__).resolve().parents[1]
    td_path = Path(args.task_descriptions) if args.task_descriptions else (repo_root / "catp_base/dataset/task_descriptions.txt")
    _TASK_DESCRIPTIONS = _load_task_descriptions(td_path)

    def _to_int_nested(d: Dict[Any, Any]) -> Dict[int, Dict[int, Any]]:
        out: Dict[int, Dict[int, Any]] = {}
        for k, v in d.items():
            tk = int(k)
            if isinstance(v, dict):
                out[tk] = {int(kk): vv for kk, vv in v.items()}
            else:
                out[tk] = v
        return out

    cp: JsonCheckpointer | None = None
    if args.checkpoint_path:
        cp = JsonCheckpointer(args.checkpoint_path, args.autosave_steps)
        loaded = cp.load_if_exists()
        if not loaded:
            cp.state = {
                "inputs": {
                    "pkls": list(args.pkls),
                    "out_dir": args.out_dir,
                    "task_descriptions": args.task_descriptions,
                },
                "started_at": int(Path(args.out_dir).stat().st_mtime if Path(args.out_dir).exists() else __import__("time").time()),
                "elapsed_sec": 0.0,
                "phase": "execute",
                "processed_keys": [],
                "merged_valid_best": {},
                "merged_invalid_best": {},
                "merged_valid_all": {},
                "merged_invalid_all": {},
            }
            cp.save()

    if cp is not None and cp.state.get("merged_valid_best") is not None:
        merged_valid_best: Dict[int, Dict[int, Any]] = _to_int_nested(cp.state.get("merged_valid_best", {}))
        merged_invalid_best: Dict[int, Dict[int, Any]] = _to_int_nested(cp.state.get("merged_invalid_best", {}))
        merged_valid_all: Dict[int, Dict[int, List[Any]]] = _to_int_nested(cp.state.get("merged_valid_all", {}))  # type: ignore
        merged_invalid_all: Dict[int, Dict[int, List[Any]]] = _to_int_nested(cp.state.get("merged_invalid_all", {}))  # type: ignore
        processed_keys: List[str] = list(cp.state.get("processed_keys", []))
    else:
        merged_valid_best = {}
        merged_invalid_best = {}
        merged_valid_all = {}
        merged_invalid_all = {}
        processed_keys = []

    processed_set = set(processed_keys)

    for pkl_path in args.pkls:
        obj = pickle.load(open(pkl_path, "rb"))
        plan_pools = obj if isinstance(obj, list) else [obj]
        for plan_pool in plan_pools:
            for task_id, samples in plan_pool.plans.items():
                tid = int(task_id)
                for sample_id in samples.keys():
                    sid = int(sample_id)
                    plans = plan_pool.plans[task_id][sample_id]
                    plan_list = [plans] if isinstance(plans, tuple) else plans
                    for idx, plan in enumerate(plan_list):
                        key = f"{pkl_path}|{tid}|{sid}|{int(idx)}"
                        if key in processed_set:
                            continue
                        plan_exec = _resolve_prev_deps(plan)
                        res = _run_plan(tid, sid, plan_exec)
                        if not res.get("valid"):
                            invalid_variants = merged_invalid_all.setdefault(tid, {}).setdefault(sid, [])
                            invalid_variants.append({
                                "plan": _stringify_plan(plan),
                                "task_score": GlobalMetricsConfig.score_penalty,
                                "cost_price": GlobalMetricsConfig.cost_penalty,
                                "exec_time": None,
                                "qop": None,
                                "task_query": _task_query_for(tid),
                            })
                        else:
                            valid_variants = merged_valid_all.setdefault(tid, {}).setdefault(sid, [])
                            valid_variants.append({
                                "plan": _stringify_plan(plan),
                                "task_score": res["task_score"],
                                "cost_price": res["cost_price"],
                                "exec_time": res["exec_time"],
                                "qop": res["qop"],
                                "task_query": _task_query_for(tid),
                            })

                        if len(merged_valid_all.get(tid, {}).get(sid, [])) > 0:
                            valid_list = merged_valid_all[tid][sid]
                            best = max(valid_list, key=lambda x: x["qop"])
                            merged_valid_best.setdefault(tid, {})[sid] = best
                            if tid in merged_invalid_best and sid in merged_invalid_best[tid]:
                                del merged_invalid_best[tid][sid]
                                if not merged_invalid_best[tid]:
                                    del merged_invalid_best[tid]
                        elif len(merged_invalid_all.get(tid, {}).get(sid, [])) > 0 and sid not in merged_invalid_best.get(tid, {}):
                            worst = merged_invalid_all[tid][sid][0]
                            merged_invalid_best.setdefault(tid, {})[sid] = worst

                        processed_keys.append(key)
                        processed_set.add(key)

                        if cp is not None:
                            cp.state["merged_valid_best"] = merged_valid_best
                            cp.state["merged_invalid_best"] = merged_invalid_best
                            cp.state["merged_valid_all"] = merged_valid_all
                            cp.state["merged_invalid_all"] = merged_invalid_all
                            cp.state["processed_keys"] = processed_keys
                            cp.step()

    with open(os.path.join(args.out_dir, "valid_plans_best.json"), "w", encoding="utf-8") as f:
        json.dump(merged_valid_best, f, indent=2)
    with open(os.path.join(args.out_dir, "invalid_plans_best.json"), "w", encoding="utf-8") as f:
        json.dump(merged_invalid_best, f, indent=2)
    with open(os.path.join(args.out_dir, "valid_plans_all.json"), "w", encoding="utf-8") as f:
        json.dump(merged_valid_all, f, indent=2)
    with open(os.path.join(args.out_dir, "invalid_plans_all.json"), "w", encoding="utf-8") as f:
        json.dump(merged_invalid_all, f, indent=2)

    if cp is not None:
        cp.tick()
        cp.save()


if __name__ == "__main__":
    main()


