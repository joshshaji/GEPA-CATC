import argparse
import json
import os
import sys
from glob import glob


def iterate_objects(value):
    if isinstance(value, dict):
        if "plan" in value:
            yield value
        for child in value.values():
            yield from iterate_objects(child)
    elif isinstance(value, list):
        for item in value:
            yield from iterate_objects(item)


def count_samples_in_data(data):
    if isinstance(data, dict):
        total = 0
        for value in data.values():
            if isinstance(value, dict):
                total += sum(1 for v in value.values())
            elif isinstance(value, list):
                total += len(value)
        return total
    if isinstance(data, list):
        return sum(count_samples_in_data(item) if isinstance(item, (dict, list)) else 0 for item in data)
    return sum(1 for _ in iterate_objects(data))


def collect_json_files(paths):
    collected = []
    for path in paths:
        expanded = glob(path) or [path]
        for candidate in expanded:
            if os.path.isdir(candidate):
                for dirpath, _, filenames in os.walk(candidate):
                    for filename in filenames:
                        if filename.lower().endswith(".json"):
                            collected.append(os.path.join(dirpath, filename))
            else:
                if candidate.lower().endswith(".json") and os.path.exists(candidate):
                    collected.append(candidate)
    unique = sorted(set(os.path.abspath(p) for p in collected))
    return unique


def main():
    parser = argparse.ArgumentParser(prog="count_plans", description="Count number of samples (task_id -> sample_id) in JSON files.")
    parser.add_argument("paths", nargs="*", help="JSON files, directories, or globs. Defaults to seq_data/valid_plans_*.json if omitted.")
    args = parser.parse_args()

    default_paths = [
        "seq_data/valid_plans_best.json",
        "non_seq_data/valid_plans_best.json",
    ]

    target_paths = args.paths if args.paths else [p for p in default_paths if os.path.exists(p)]
    if not target_paths:
        print("No input provided and no default plan files found.", file=sys.stderr)
        sys.exit(1)

    files = collect_json_files(target_paths)
    if not files:
        print("No JSON files matched the provided inputs.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = count_samples_in_data(data)
            print(f"{file_path}: {count}")
            total += count
        except Exception as exc:
            print(f"{file_path}: error: {exc}", file=sys.stderr)

    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()


