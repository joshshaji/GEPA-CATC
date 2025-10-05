#!/usr/bin/env python3
"""
Convert plans_dag_loss.json into a nested mapping:

{
    "<task_id>": {
        "<sample_id>": {
            "plan": "[tool1, [deps], tool2, [deps], ...]"
        },
        ...
    },
    ...
}

By default, uses the 'pred_plan' field. You can switch to 'gold_plan'
via CLI flag.

Usage:
  python scripts/convert_plans_dag_loss.py \
      --input plans_dag_loss.json \
      --output plans_dag_loss_converted.json \
      --plan-key pred   # or gold
"""

import argparse
import json
import sys
from collections import defaultdict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert plans_dag_loss.json to nested {task_id: {sample_id: {plan: str}}} format")
    p.add_argument(
        "--input", "-i", default="plans_dag_loss.json",
        help="Path to input JSON file (default: plans_dag_loss.json)"
    )
    p.add_argument(
        "--output", "-o", default="plans_dag_loss_converted.json",
        help="Path to output JSON file (default: plans_dag_loss_converted.json)"
    )
    p.add_argument(
        "--plan-key", "-k", choices=["pred", "gold"], default="pred",
        help="Which plan field to use: 'pred' -> pred_plan, 'gold' -> gold_plan (default: pred)"
    )
    p.add_argument(
        "--sort-samples", action="store_true",
        help="Sort inner sample_id keys numerically for stable output"
    )
    return p


def main():
    args = build_parser().parse_args()

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    items = data.get("items")
    if not isinstance(items, list):
        print("[ERROR] Input JSON missing 'items' list.", file=sys.stderr)
        sys.exit(1)

    # Map CLI plan-key to field name in input
    plan_field = "pred_plan" if args.plan_key == "pred" else "gold_plan"

    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    seen = set()
    duplicate_count = 0
    missing_plan_count = 0

    for rec in items:
        task_id = rec.get("task_id")
        sample_id = rec.get("sample_id")
        plan = rec.get(plan_field)

        if task_id is None or sample_id is None:
            # skip malformed records
            continue

        key = (task_id, sample_id)
        if key in seen:
            duplicate_count += 1
        seen.add(key)

        if plan is None:
            missing_plan_count += 1
            continue

        # Represent plan as a string (single-quoted Python-style list),
        # consistent with existing dataset files in repo.
        plan_str = repr(plan)

        out[str(task_id)][str(sample_id)] = {"plan": plan_str}

    # Optionally sort the inner sample_id keys for stable output
    if args.sort_samples:
        sorted_out: dict[str, dict[str, dict[str, str]]] = {}
        for t_id, samples in out.items():
            # sort numerically when possible
            try:
                items_sorted = dict(sorted(samples.items(), key=lambda kv: int(kv[0])))
            except Exception:
                items_sorted = dict(sorted(samples.items(), key=lambda kv: kv[0]))
            sorted_out[t_id] = items_sorted
        out = sorted_out

    with open(args.output, "w") as f:
        json.dump(out, f, indent=4)

    msg = [
        f"Converted {len(items)} records.",
        f"Output written to: {args.output}",
    ]
    if duplicate_count:
        msg.append(f"Note: encountered {duplicate_count} duplicate (task_id, sample_id) pairs; last occurrence kept.")
    if missing_plan_count:
        msg.append(f"Note: skipped {missing_plan_count} records missing '{plan_field}'.")
    print(" ".join(msg))


if __name__ == "__main__":
    main()

