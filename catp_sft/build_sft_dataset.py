#!/usr/bin/env python3
"""Utility to shrink GRPO datasets down to the fields needed for SFT."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


def _prune_task_payload(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only task_query, image_size, and gold_plan for every image."""
    pruned_images: Dict[str, Dict[str, Any]] = {}
    for image_id, image_payload in task_payload.get("images", {}).items():
        pruned_images[image_id] = {
            "image_size": image_payload.get("image_size"),
            "gold_plan": image_payload.get("gold_plan"),
        }
    return {
        "task_query": task_payload.get("task_query"),
        "images": pruned_images,
    }


def convert_dataset(input_path: Path, output_path: Path) -> None:
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    pruned = {task_id: _prune_task_payload(task_payload) for task_id, task_payload in dataset.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    logger.info("Wrote %d tasks to %s", len(pruned), output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract minimal fields for SFT from GRPO dataset.")
    parser.add_argument("--input", required=True, help="Path to grpo_dataset_*.json input.")
    parser.add_argument("--output", required=True, help="Destination path for the pruned JSON.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    convert_dataset(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
