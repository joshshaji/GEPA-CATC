from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Callable

import dspy
import time
from dspy.adapters.chat_adapter import ChatAdapter

from catp_gepa.run_state import (
    RunState,
    RunEvent,
    PredictorCall,
    summarize_prediction,
    extract_example_fields,
)

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
        q = float(v.get("qop", -0.055))
        if q > best_qop:
            best_qop = q
            best_pairs = _normalize_pairs(v.get("plan"))
    return best_qop if best_qop != float("-inf") else -0.055, best_pairs


def metric_qop(example: dspy.Example, prediction: dspy.Prediction, trace: object | None = None, *_, **__) -> float:
    variants = _load_variants(example)
    gen_pairs = _normalize_pairs(prediction["plan_json"]) if "plan_json" in prediction else _normalize_pairs(prediction.plan_json)
    for v in variants:
        if _normalize_pairs(v.get("plan")) == gen_pairs:
            return float(v.get("qop", -0.055))
    return -0.055


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



def vanila_gepa_metric(example: dspy.Example, prediction: dspy.Prediction, trace: object | None = None, *_, **__)  -> dspy.Prediction:
    """
    A simple, QOP-free, structure-aware metric for benchmarking.

    - Parses predicted and gold plans.
    - Computes two F1 scores:
        1) tool-set F1 (unique tools used)
        2) dependency-edge F1 over directed edges (dep -> tool), excluding the
           synthetic "input_of_query" edges to focus on inter-tool structure.
    - Final score is the mean of the two F1s. This keeps it simple while still
      rewarding correct tools and correct wiring between them.
    - Feedback lists missing/extra tools and missing/extra edges.
    """
    # Extract the gold plan and the predicted plan
    try:
        gold_plan_json = example["gold_plan_json"] if "gold_plan_json" in example else getattr(example, "gold_plan_json", "")
        task_query = example["task_query"] if "task_query" in example else getattr(example, "task_query", "")
    except Exception:
        logger.exception("Missing required fields in example for vanilla metric")
        return dspy.Prediction(score=0.0, feedback="Missing required fields in example.")

    pred_plan_json = prediction["plan_json"] if "plan_json" in prediction else getattr(prediction, "plan_json", "")

    # Normalize into pairs [tool, deps]
    gold_pairs = _normalize_pairs(gold_plan_json)
    pred_pairs = _normalize_pairs(pred_plan_json)

    # Helpers to collect unique tool names and dependency edges
    def tool_set(pairs: List[List[Any]]) -> set:
        return {str(p[0]) for p in pairs if isinstance(p, list) and p}

    def edge_set(pairs: List[List[Any]]) -> set:
        edges = set()
        for p in pairs:
            if not isinstance(p, list) or not p:
                continue
            tool = str(p[0])
            deps = p[1] if len(p) > 1 and isinstance(p[1], list) else []
            for dep in deps:
                dep_s = str(dep)
                if dep_s == "input_of_query":
                    continue
                edges.add((dep_s, tool))
        return edges

    gold_tools = tool_set(gold_pairs)
    pred_tools = tool_set(pred_pairs)

    gold_edges = edge_set(gold_pairs)
    pred_edges = edge_set(pred_pairs)

    # Compute F1 helpers
    def f1(gold: set, pred: set) -> float:
        if not gold and not pred:
            return 1.0
        if not gold or not pred:
            return 0.0
        tp = len(gold & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    tool_f1 = f1(gold_tools, pred_tools)
    edge_f1 = f1(gold_edges, pred_edges)
    score = 0.5 * tool_f1 + 0.5 * edge_f1

    # Build concise feedback for reflection without referencing QOP
    missing = sorted(list(gold_tools - pred_tools))
    extra = sorted(list(pred_tools - gold_tools))
    # Build feedback
    feedback_lines = [
        f"User query: {task_query}",
        f"Tool F1: {tool_f1:.3f} | Edge F1: {edge_f1:.3f}",
        f"Predicted tools: {sorted(list(pred_tools))}",
        f"Gold tools: {sorted(list(gold_tools))}",
    ]
    if missing:
        feedback_lines.append(f"Missing tools to include: {missing}")
    if extra:
        feedback_lines.append(f"Extra tools to remove: {extra}")
    if not missing and not extra:
        feedback_lines.append("Tool sets match.")

    # Edge-level feedback
    missing_edges = sorted(list(gold_edges - pred_edges))
    extra_edges = sorted(list(pred_edges - gold_edges))
    if missing_edges:
        feedback_lines.append(
            "Missing edges (dep -> tool): " + ", ".join([f"{a} -> {b}" for a, b in missing_edges])
        )
    if extra_edges:
        feedback_lines.append(
            "Extra edges (dep -> tool): " + ", ".join([f"{a} -> {b}" for a, b in extra_edges])
        )
    if not missing_edges and not extra_edges:
        feedback_lines.append("Dependency edges match.")

    feedback = "\n".join(feedback_lines)
    return dspy.Prediction(score=float(score), feedback=feedback)


def make_logged_metric(
    metric_fn: Callable[[dspy.Example, Any, Optional[object]], Any],
    run_state: RunState,
    stage_label: str,
    *,
    adapter_for_prompt: Optional[ChatAdapter] = None,
):
    """Wrap a metric function to record a RunEvent into run_state.

    - Calls the underlying metric to get its result (float, dict, or Prediction).
    - Extracts score/feedback when possible for convenience.
    - Captures prompt messages and raw completions from the provided trace.
    - Includes task_and_sample_id derived from the example when available.
    """
    adapter = adapter_for_prompt or ChatAdapter()

    def _wrapped(example, prediction, trace=None, *args, **kwargs):
        # Compute the metric first
        result = metric_fn(example, prediction, trace, *args, **kwargs)

        # Derive score/feedback
        score = None
        feedback = None
        try:
            if isinstance(result, dict):
                # Metric returned a mapping
                score = result.get("score")
                feedback = result.get("feedback")
            elif hasattr(result, "score") or hasattr(result, "feedback"):
                # Metric returned a dspy.Prediction-like object
                try:
                    s = getattr(result, "score", None)
                    score = float(s) if s is not None else None
                except Exception:
                    # Leave score as None if it can't be coerced
                    pass
                try:
                    feedback = getattr(result, "feedback", None)
                except Exception:
                    # Leave feedback as None if not accessible
                    pass
            else:
                # Metric returned a scalar (e.g., float or int)
                score = float(result)
        except Exception:
            # Never let logging extraction break the run
            pass

        # Build event
        try:
            tid = getattr(example, "task_id", None)
            sid = getattr(example, "sample_id", None)
        except Exception:
            tid, sid = None, None

        event = RunEvent(
            when=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            stage=stage_label,
            task_and_sample_id=(f"{tid}:{sid}" if tid is not None and sid is not None else "unknown"),
            example_fields=extract_example_fields(example),
            prediction_summary=summarize_prediction(prediction),
            score=score,
            feedback=feedback,
            predictor_calls=[],
            error=None,
        )

        # Attach predictor-level info if trace is available
        try:
            if trace:
                for pred_obj, pred_inputs, pred_outputs in trace:
                    try:
                        messages = adapter.format(
                            signature=pred_obj.signature,
                            demos=getattr(pred_obj, "demos", []),
                            inputs=pred_inputs,
                        )
                    except Exception:
                        messages = []

                    raw_failure = None
                    try:
                        raw_failure = getattr(pred_outputs, "completion_text", None)
                    except Exception:
                        raw_failure = None

                    call = PredictorCall(
                        when=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        predictor_id=getattr(pred_obj, "stage", None),
                        predictor_name=None,
                        inputs=pred_inputs,
                        outputs=summarize_prediction(pred_outputs),
                        messages=messages,
                        raw_completion_on_failure=raw_failure,
                    )
                    event.predictor_calls.append(call)
        except Exception:
            # Don't let logging break the run
            pass

        run_state.add_event(event)
        return result

    return _wrapped
