import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PlanEntry = Dict[str, float | str]
Plans = Dict[str, Dict[str, List[PlanEntry]]]
TransformedPlans = Dict[str, Dict[str, object]]


def select_gold_plan(entries: List[PlanEntry]) -> Tuple[PlanEntry, List[PlanEntry]]:
    def slim(entry: PlanEntry) -> PlanEntry:
        return {
            key: entry[key]
            for key in ("plan", "qop", "task_score", "cost_price")
            if key in entry
        }

    trimmed = [slim(entry) for entry in entries]
    gold = max(trimmed, key=lambda item: item.get("qop", float("-inf")))
    return gold, trimmed


def load_task_queries(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def resolve_image_path(dataset_root: Path, task_id: str, image_id: str) -> Path:
    candidates: Iterable[Path] = [
        dataset_root / task_id / "inputs" / "images" / f"{image_id}{suffix}"
        for suffix in (".jpg", ".png", ".jpeg", ".bmp")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find image for task {task_id} image {image_id} within {dataset_root}."
    )


def _read_jpeg_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        handle.seek(0)
        if handle.read(2) != b"\xFF\xD8":
            raise ValueError(f"File {path} is not a valid JPEG.")
        while True:
            marker_head = handle.read(2)
            while marker_head and marker_head[0] != 0xFF:
                marker_head = marker_head[1:] + handle.read(1)
            if not marker_head:
                break
            marker = marker_head[1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                handle.read(3)
                height, width = struct.unpack(">HH", handle.read(4))
                return width, height
            else:
                segment_length_bytes = handle.read(2)
                if len(segment_length_bytes) != 2:
                    break
                segment_length = struct.unpack(">H", segment_length_bytes)[0]
                handle.seek(segment_length - 2, 1)
    raise ValueError(f"Could not determine JPEG size for {path}.")


def _read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"File {path} is not a valid PNG.")
        chunk_header = handle.read(8)
        if len(chunk_header) != 8 or chunk_header[4:] != b"IHDR":
            raise ValueError(f"File {path} missing IHDR chunk.")
        width, height = struct.unpack(">II", handle.read(8))
        return width, height


def _read_bmp_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(26)
        if len(header) < 26 or header[:2] != b"BM":
            raise ValueError(f"File {path} is not a valid BMP.")
        width = struct.unpack("<I", header[18:22])[0]
        height = struct.unpack("<I", header[22:26])[0]
        return width, height


def fetch_image_size(dataset_root: Path, task_id: str, image_id: str) -> Tuple[int, int]:
    image_path = resolve_image_path(dataset_root, task_id, image_id)
    suffix = image_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return _read_jpeg_size(image_path)
    if suffix == ".png":
        return _read_png_size(image_path)
    if suffix == ".bmp":
        return _read_bmp_size(image_path)
    raise ValueError(f"Unsupported image type for size extraction: {image_path}")


def transform_plans(
    tasks: Plans, task_queries: List[str], dataset_root: Path
) -> TransformedPlans:
    transformed: TransformedPlans = {}
    for task_id, images in tasks.items():
        idx = int(task_id)
        if idx < 0 or idx >= len(task_queries):
            raise IndexError(
                f"Task id {task_id} out of bounds for task descriptions (size={len(task_queries)})."
            )
        transformed[task_id] = {
            "task_query": task_queries[idx],
            "images": {},
        }
        for image_id, entries in images.items():
            gold_plan, all_plans = select_gold_plan(entries)
            width, height = fetch_image_size(dataset_root, task_id, image_id)
            transformed[task_id]["images"][image_id] = {
                "image_size": [width, height],
                "gold_plan": gold_plan,
                "plans": all_plans,
            }
    return transformed


def load_json(path: Path) -> Plans:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: TransformedPlans) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate plans by task/image and annotate the gold plan (max QoP)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the transformed JSON payload.",
    )
    parser.add_argument(
        "--dataset-root",
        default="catp-llm/dataset",
        help="Root directory containing task folders and inputs/images.",
    )
    parser.add_argument(
        "--task-descriptions",
        default=None,
        help="Optional path to task_descriptions.txt. Defaults to <dataset-root>/task_descriptions.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    dataset_root = Path(args.dataset_root)
    descriptions_path = (
        Path(args.task_descriptions)
        if args.task_descriptions
        else dataset_root / "task_descriptions.txt"
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not descriptions_path.exists():
        raise FileNotFoundError(f"Task descriptions not found: {descriptions_path}")

    task_queries = load_task_queries(descriptions_path)
    tasks = load_json(input_path)
    transformed = transform_plans(tasks, task_queries, dataset_root)
    save_json(output_path, transformed)


if __name__ == "__main__":
    main()
