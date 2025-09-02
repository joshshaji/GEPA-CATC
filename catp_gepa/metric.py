from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import dspy

logger = logging.getLogger(__name__)


def _parse_plan(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            try:
                import ast
                return ast.literal_eval(obj)
            except Exception:
                logger.warning("Parse failed; returning empty plan")
                return []
    return obj


def _to_pairs(plan_obj: Any) -> List[Tuple[str, List[str]]]:
    plan_obj = _parse_plan(plan_obj)
    if isinstance(plan_obj, dict) and "plan" in plan_obj:
        plan_obj = plan_obj["plan"]
    if isinstance(plan_obj, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in plan_obj):
        out: List[Tuple[str, List[str]]] = []
        for tool, deps in plan_obj:
            out.append((str(tool), list(deps) if isinstance(deps, list) else []))
        return out
    if isinstance(plan_obj, list) and all(isinstance(x, (str, list)) for x in plan_obj):
        out: List[Tuple[str, List[str]]] = []
        i = 0
        n = len(plan_obj)
        while i + 1 < n:
            tool = plan_obj[i]
            deps = plan_obj[i + 1]
            if isinstance(tool, str) and isinstance(deps, list):
                out.append((tool, deps))
            i += 2
        return out
    return []


def _normalize_pairs(plan_obj: Any) -> List[List[Any]]:
    pairs = _to_pairs(plan_obj)
    if not pairs:
        return []
    result: List[List[Any]] = []
    for idx, (tool, deps) in enumerate(pairs):
        deps_list = deps if isinstance(deps, list) else []
        if idx == 0 and not deps_list:
            deps_list = ["input_of_query"]
        result.append([tool, deps_list])
    return result


def _flatten(pairs: List[List[Any]]) -> List[Any]:
    flat: List[Any] = []
    for tool, deps in pairs:
        flat.append(tool)
        flat.append(deps if isinstance(deps, list) else [])
    return flat


def _load_variants(example: dspy.Example) -> List[Dict[str, Any]]:
    return json.loads(example["plan_variants_json"]) if "plan_variants_json" in example else json.loads(getattr(example, "plan_variants_json", "[]"))


def _best_variant(variants: List[Dict[str, Any]]) -> Tuple[float, List[List[Any]]]:
    best_qop = float("-inf")
    best_pairs: List[List[Any]] = []
    for v in variants:
        q = float(v.get("qop", -2.0))
        if q > best_qop:
            best_qop = q
            best_pairs = _normalize_pairs(v.get("plan"))
    return best_qop if best_qop != float("-inf") else -2.0, best_pairs


def metric_qop(example: dspy.Example, prediction: dspy.Prediction, trace: object | None = None, *_, **__) -> float:
    variants = _load_variants(example)
    gen_pairs = _normalize_pairs(prediction["plan_json"]) if "plan_json" in prediction else _normalize_pairs(prediction.plan_json)
    for v in variants:
        if _normalize_pairs(v.get("plan")) == gen_pairs:
            return float(v.get("qop", -2.0))
    return -2.0


def metric_qop_feedback(example: dspy.Example, prediction: dspy.Prediction, trace: object | None = None, *_, **__) -> dspy.Prediction:
    score = metric_qop(example, prediction)
    variants = _load_variants(example)
    best_qop, best_pairs = _best_variant(variants)
    gold_plan_json = example["gold_plan_json"] if "gold_plan_json" in example else getattr(example, "gold_plan_json", "")

    try:
        task_query = example["task_query"] if "task_query" in example else getattr(example, "task_query", "")
    except Exception:
        logger.exception("No task query found in example")
        raise ValueError("No task query found in example")

    if score < 0:
        feedback = f"""
        The user query is: {task_query}
        {example['tool_catalog_json_with_description']}
        The input attributes for the current task are: {example.get('input_attributes_json', '{}')}
        The generated plan is: {prediction['plan_json'] if 'plan_json' in prediction else prediction.plan_json}
        An invalid or unrecognized plan was generated for this task. Plan not found in known variants; This means that the generated plan is below the worst variant.
        The best plan (gold plan) for this task is: {gold_plan_json} which is both cost and accuracy optimal.
        """
        return dspy.Prediction(score=score, feedback=feedback)

    elif score == best_qop:
        feedback = f"""
        The user query is: {task_query}
        {example['tool_catalog_json_with_description']}
        The input attributes for the current task are: {example.get('input_attributes_json', '{}')}
        The generated plan is: {prediction['plan_json'] if 'plan_json' in prediction else prediction.plan_json}
        The generated plan is the best plan for this task.
        This means that the generated plan is both cost and accuracy optimal.
        """
        return dspy.Prediction(score=score, feedback=feedback)

    else:
        feedback = f"""
        The user query is: {task_query}
        {example['tool_catalog_json_with_description']}
        The input attributes for the current task are: {example.get('input_attributes_json', '{}')}
        The generated plan is: {prediction['plan_json'] if 'plan_json' in prediction else prediction.plan_json}
        The generated plan is NOT THE MOST OPTIMAL PLAN for this task.
        The best plan (gold plan) for this task is: {gold_plan_json} which is both cost and accuracy optimal.
        """
        return dspy.Prediction(score=score, feedback=feedback)

