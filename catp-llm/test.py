"""
Evaluate CATP plan outputs by executing each plan and computing QoP metrics.

Usage:
    python catp-llm/test.py \
        --input-json catp_results/llama37B_nonseq_gepa.json \
        --output-json catp_results/results_llama37B_nonseq_gepa.json

The script iterates over every task/image pair in the input JSON, executes the
plan on the corresponding dataset sample, and stores task score, QoP, and cost
information in the output JSON. Summary statistics are printed to stdout.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader

from src.config import GlobalPathConfig
from src.data_loader import TaskDataset
from src.metrics import calculate_qop, calculate_task_score
from src.plan import Plan


def get_task_results(task_id: str, img_id: str, plan_str: str) -> Optional[Dict]:
    """
    Execute a single plan and compute QoP metrics for the specified sample.

    Returns None if execution fails or the sample cannot be found.
    """
    try:
        plan_list = ast.literal_eval(plan_str)
    except (SyntaxError, ValueError) as exc:
        print(f"Failed to parse plan for task {task_id}, image {img_id}: {exc}")
        return None

    plan = Plan(plan_list)
    task_id_int = int(task_id)

    data_set = TaskDataset(GlobalPathConfig.data_path, task_id=task_id_int)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    for batch in data_loader:
        sample_id = batch["sample_id"]
        if int(sample_id.item()) != int(img_id):
            continue

        input_data = batch["input"]
        output_data = batch["output"]

        try:
            result = plan.execute(input_data)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Plan execution failed for task {task_id}, image {img_id}: {exc}")
            return None

        if result is None:
            print(f"Plan returned None for task {task_id}, image {img_id}")
            return None

        cost_price = plan.price
        exec_time = plan.exec_time
        task_score = calculate_task_score(result, output_data, sequential=task_id_int < 200)
        qop = calculate_qop(task_score, cost_price)

        print(
            f"\nTask {task_id} | Image {img_id} | QoP {qop:.4f} | "
            f"Task Score {task_score:.4f} | Cost Price {cost_price:.4f}\n"
        )
        return {
            "task_id": task_id,
            "image_id": img_id,
            "qop": qop,
            "task_score": task_score,
            "cost_price": cost_price,
            "exec_time": exec_time,
        }

    print(f"No matching sample found for task {task_id}, image {img_id}")
    return None


def evaluate_plans(input_path: Path, output_path: Path) -> None:
    """Evaluate every plan in the input JSON and write augmented results."""
    data = json.loads(input_path.read_text(encoding="utf-8"))

    output_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    failed_tasks = []
    qop_sum = 0.0
    success_count = 0

    for task_id, images in data.items():
        output_data[task_id] = {}

        for img_id, payload in images.items():
            plan_str = payload.get("plan", "")
            task_result = get_task_results(task_id, img_id, plan_str)

            if task_result is None:
                failed_tasks.append({"task_id": task_id, "image_id": img_id})
                continue

            output_data[task_id][img_id] = {
                "plan": payload.get("plan"),
                "qop": task_result["qop"],
                "task_score": task_result["task_score"],
                "cost_price": task_result["cost_price"],
            }
            qop_sum += task_result["qop"]
            success_count += 1

    average_qop = qop_sum / success_count if success_count else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print(f"Results saved to {output_path}")
    print(f"Average QoP of the input JSON: {average_qop:.6f}")
    print(f"Total successful tasks: {success_count}")
    print(f"Total failed tasks: {len(failed_tasks)}")

    if failed_tasks:
        print("\nFailed tasks:")
        for entry in failed_tasks:
            print(f"  Task ID: {entry['task_id']}, Image ID: {entry['image_id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CATP plan JSON outputs.")
    parser.add_argument("--input-json", required=True, help="Path to the JSON file containing generated plans.")
    parser.add_argument("--output-json", required=True, help="Path where the augmented results JSON will be written.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser()
    output_path = Path(args.output_json).expanduser()
    evaluate_plans(input_path, output_path)


if __name__ == "__main__":
    main()
