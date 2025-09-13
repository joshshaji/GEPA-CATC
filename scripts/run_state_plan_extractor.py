from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List, Optional


def _parse_plan_field(v: Any) -> Optional[List[Any]]:
    """Best-effort parse of a plan field which may be a list or a stringified list."""
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None
    return None


def _extract_task_sample(ts: str) -> Optional[tuple[int, int]]:
    """Parse task_and_sample_id formatted as 'task:sample'."""
    if not ts or ts == "unknown":
        return None
    parts = str(ts).split(":", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def extract_from_run_state(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    events = data.get("events", []) if isinstance(data, dict) else []
    for ev in events:
        try:
            if ev.get("stage") != "post_eval":
                continue
            ts = _extract_task_sample(ev.get("task_and_sample_id"))
            if ts is None:
                continue
            task_id, sample_id = ts

            example_fields = ev.get("example_fields", {}) or {}
            gold_plan_raw = example_fields.get("gold_plan_json")
            gold_plan = _parse_plan_field(gold_plan_raw)

            pred_summary = ev.get("prediction_summary", {}) or {}
            pred_plan = pred_summary.get("plan_json")

            if gold_plan is None or pred_plan is None:
                continue

            out.append({
                "task_id": task_id,
                "sample_id": sample_id,
                "gold_plan": gold_plan,
                "pred_plan": pred_plan,
                "source_file": os.path.basename(path),
            })
        except Exception:
            # Skip malformed events but continue processing others
            continue
    return out


def find_default_inputs() -> List[str]:
    candidates = []
    # Prefer success logs first, then any run_state_*.json
    candidates.extend(sorted(glob(os.path.join("gepa_logs", "run_state_success_*.json"))))
    candidates.extend(sorted(glob(os.path.join("gepa_logs", "run_state_*.json"))))
    return candidates


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract gold and predicted plans from post_eval events in run_state JSON logs.")
    parser.add_argument("inputs", nargs="*", help="Paths to run_state JSON files or directories. Defaults to gepa_logs.")
    parser.add_argument("-o", "--output", help="Output JSON path. Defaults to stdout.")
    args = parser.parse_args(argv)

    inputs: List[str] = []
    if args.inputs:
        for p in args.inputs:
            if os.path.isdir(p):
                inputs.extend(sorted(glob(os.path.join(p, "*.json"))))
            else:
                inputs.append(p)
    else:
        inputs = find_default_inputs()

    if not inputs:
        sys.stderr.write("No input run_state JSON files found.\n")
        return 2

    records: List[Dict[str, Any]] = []
    for path in inputs:
        try:
            recs = extract_from_run_state(path)
            records.extend(recs)
        except Exception as e:
            sys.stderr.write(f"Failed to process {path}: {e}\n")

    output_obj = {
        "count": len(records),
        "items": records,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)
    else:
        json.dump(output_obj, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

