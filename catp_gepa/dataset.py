from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dspy
try:
    from src.config import GlobalToolConfig
except Exception:
    import sys
    repo_root = Path(__file__).parent.parent
    catpllmdir = repo_root / "catp-llm"
    if str(catpllmdir) not in sys.path:
        sys.path.insert(0, str(catpllmdir))
    from src.config import GlobalToolConfig
from pydantic import BaseModel, ValidationError, field_validator, RootModel, model_validator


class PlanVariant(BaseModel):
    plan: List[object]
    task_score: float
    cost_price: float
    exec_time: Optional[float] = None
    qop: Optional[float] = None
    task_query: Optional[str] = None

    @field_validator("plan", mode="before")
    @classmethod
    def parse_plan(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                pass
        raise ValueError("Invalid plan representation")


class SampleVariants(RootModel[List[PlanVariant]]):
    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    def __getitem__(self, index: int) -> PlanVariant:
        return self.root[index]


class TaskSamples(RootModel[Dict[int, SampleVariants]]):
    @model_validator(mode="before")
    @classmethod
    def keys_to_int(cls, v):
        return {int(k): val for k, val in v.items()}

    def items(self):
        return self.root.items()

    def __getitem__(self, item: int) -> SampleVariants:
        return self.root[item]


class CATPDataset(RootModel[Dict[int, TaskSamples]]):
    @model_validator(mode="before")
    @classmethod
    def keys_to_int(cls, v):
        return {int(k): val for k, val in v.items()}

    def tasks(self) -> Dict[int, TaskSamples]:
        return self.root

    def __getitem__(self, item: int) -> TaskSamples:
        return self.root[item]


def load_catp_dataset(path: str | Path) -> CATPDataset:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return CATPDataset.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"Failed to parse {path}: {e}")


def _build_tool_catalog_json() -> str:
    tools = []
    for name, prices in GlobalToolConfig.tool_prices.items():
        io = GlobalToolConfig.tool_io_dict.get(name, ["unknown", "unknown"]) 
        deps = GlobalToolConfig.tool_dependencies.get(name, [])
        tools.append({
            "name": name,
            "input_type": io[0],
            "output_type": io[1],
            "dependencies": deps,
            "price_estimates": prices,
        })
    payload = {"tools": tools, "notes": "Prices are estimates across input scales."}
    return json.dumps(payload)


def _load_task_descriptions(repo_root: Path) -> List[str]:
    desc_path = repo_root / "catp-llm/dataset/task_descriptions.txt"
    if not desc_path.exists():
        return []
    with desc_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _task_query_for(task_id: int, desc: List[str]) -> Optional[str]:
    if 1 <= task_id <= len(desc):
        return desc[task_id - 1]
    if 0 <= task_id < len(desc):
        return desc[task_id]
    return None


def build_valid_plans_examples(
    dataset: CATPDataset,
    train_size: int,
    test_size: int,
    seed: int = 0
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    examples: List[dspy.Example] = []
    tool_catalog_json = _build_tool_catalog_json()
    repo_root = Path(__file__).parent.parent
    task_descriptions = _load_task_descriptions(repo_root)

    for task_id, task_samples in dataset.tasks().items():
        for sample_id, variants in task_samples.items():
            if len(variants) == 0:
                continue
            gold_idx = max(range(len(variants)), key=lambda i: (variants[i].qop or 0.0))
            task_query: Optional[str] = None
            plan_variants = []
            for i, v in enumerate(variants):
                if not task_query:
                    tq = getattr(v, "task_query", None)
                    if isinstance(tq, str) and tq.strip():
                        task_query = tq
                plan_variants.append({
                    "plan": v.plan,
                    "task_score": v.task_score,
                    "cost_price": v.cost_price,
                    "exec_time": v.exec_time,
                    "qop": v.qop if v.qop is not None else 0.0,
                })
            if not task_query:
                task_query = _task_query_for(task_id, task_descriptions) or ""
            plan_variants_json = json.dumps(plan_variants)
            gold_plan = variants[gold_idx].plan
            gold_qop = variants[gold_idx].qop if variants[gold_idx].qop is not None else 0.0
            gold_plan_json = json.dumps(gold_plan)
            ex = dspy.Example({
                "task_id": task_id,
                "sample_id": sample_id,
                "task_query": task_query,
                "tool_catalog_json": tool_catalog_json,
                "plan_variants_json": plan_variants_json,
                "gold_plan_json": gold_plan_json,
                "gold_qop": gold_qop,
            }).with_inputs("task_query", "tool_catalog_json")
            examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(examples)

    test_set = examples[:test_size]
    train_val = examples[test_size:]

    # Build per-task index for remaining examples
    examples_by_task: Dict[int, List[dspy.Example]] = {}
    for ex in train_val:
        tid = getattr(ex, "task_id", None)
        if tid is None:
            continue
        examples_by_task.setdefault(int(tid), []).append(ex)

    # Select up to N samples per task for training (default: 3)
    per_task_cap = 3
    task_ids = list(examples_by_task.keys())
    rng.shuffle(task_ids)
    train_set: List[dspy.Example] = []
    remaining_slots = max(0, train_size)
    for tid in task_ids:
        if remaining_slots <= 0:
            break
        bucket = examples_by_task.get(tid, [])
        if not bucket:
            continue
        take = min(per_task_cap, len(bucket), remaining_slots)
        if take > 0:
            train_set.extend(bucket[:take])
            del bucket[:take]
            remaining_slots -= take

    # Remaining examples form the validation pool
    remaining: List[dspy.Example] = []
    for bucket in examples_by_task.values():
        remaining.extend(bucket)
    # Keep order randomized
    rng.shuffle(remaining)
    val_set = remaining

    return train_set, val_set, test_set

