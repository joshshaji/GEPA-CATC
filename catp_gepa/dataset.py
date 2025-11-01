from __future__ import annotations

import ast
import json
import random
from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple, Any

import dspy
try:
    from src.config import GlobalToolConfig, GlobalPathConfig, GlobalDataConfig
    from src.catpllm.utils.utils import get_task_and_sample_info, determine_sample_size
    from src.catpllm.utils.cost_utils import determine_input_level
except Exception:
    import sys
    repo_root = Path(__file__).parent.parent
    catpllmdir = repo_root / "catp_base"
    if str(catpllmdir) not in sys.path:
        sys.path.insert(0, str(catpllmdir))
    from src.config import GlobalToolConfig, GlobalPathConfig, GlobalDataConfig
    from src.catpllm.utils.utils import get_task_and_sample_info, determine_sample_size
    from src.catpllm.utils.cost_utils import determine_input_level
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


def load_image_metrics_map(path: str | Path | None = None) -> Dict[int, Dict[int, Dict[str, Any]]]:
    default_dir = Path("/Users/joshinshaji/dev/GEPA-CATC/gepa_logs/image_diagnostics")
    p = Path(path) if path else default_dir
    if not p.exists():
        return {}
    if p.is_dir():
        out: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for j in p.rglob("*.diagnostics.json"):
            parts = j.parts
            try:
                img_idx = len(parts) - 1 - parts[::-1].index("images")
                if img_idx - 2 < 0 or parts[img_idx - 1] != "inputs":
                    continue
                task_str = parts[img_idx - 2]
                tid = int(task_str)
                name = j.name
                if not name.endswith(".diagnostics.json"):
                    continue
                sample_str = name[: -len(".diagnostics.json")]
                sid = int(sample_str)
            except Exception:
                continue
            try:
                data = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.setdefault(tid, {})[sid] = data
        return out
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for tid_str, samples in raw.items():
        try:
            tid = int(tid_str)
        except Exception:
            continue
        out[tid] = {}
        for sid_str, variants in (samples or {}).items():
            try:
                sid = int(sid_str)
            except Exception:
                continue
            if isinstance(variants, list) and variants:
                m = variants[0].get("image_metrics")
                if isinstance(m, dict):
                    out[tid][sid] = m
    return out


def _select_relevant_image_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    res = m.get("resolution", {})
    color = m.get("color", {})
    q = m.get("quality", {})
    mb = q.get("motion_blur", {}) if isinstance(q.get("motion_blur"), dict) else {}
    return {
        "width_px": res.get("width_px"),
        "height_px": res.get("height_px"),
        "is_grayscale": color.get("is_grayscale"),
        "blur_vol": q.get("blur_vol"),
        "blur_normalized_0_1": q.get("blur_normalized_0_1"),
        "tenengrad": q.get("tenengrad"),
        "noise_sigma_luma": q.get("noise_sigma_luma"),
        "motion_blur": {
            "angle_deg": (mb or {}).get("angle_deg"),
            "confidence": (mb or {}).get("confidence"),
        },
        "blockiness": q.get("blockiness"),
        "ringing": q.get("ringing"),
        "percent_clipped_shadows": q.get("percent_clipped_shadows"),
        "percent_clipped_highlights": q.get("percent_clipped_highlights"),
    }


def _derive_targets(task_query: str, m: Dict[str, Any]) -> Dict[str, Any]:
    w = m.get("width_px") or 0
    h = m.get("height_px") or 0
    shorter = min(w, h) if (w and h) else 0
    blur_norm = m.get("blur_normalized_0_1")
    noise = m.get("noise_sigma_luma") or 0.0
    mb_conf = (m.get("motion_blur") or {}).get("confidence") or 0.0
    is_gray = bool(m.get("is_grayscale"))
    has_blur = (isinstance(blur_norm, (int, float)) and blur_norm >= 0.4)
    has_noise = noise >= 2.5
    low_res = bool(shorter and shorter < 256)
    has_motion = mb_conf >= 0.5
    rec: List[str] = []
    if is_gray:
        rec.append("image_colorization")
    if has_blur or has_motion:
        rec.append("image_deblurring")
    if has_noise:
        rec.append("image_denoising")
    if low_res:
        rec.append("image_super_resolution")
    return {
        "flags": {
            "has_blur": bool(has_blur),
            "has_noise": bool(has_noise),
            "low_resolution": bool(low_res),
            "is_grayscale": is_gray,
            "has_motion_blur": bool(has_motion),
        },
        "severity": {
            "blur": float(blur_norm) if isinstance(blur_norm, (int, float)) else None,
            "noise": float(min(noise / 10.0, 1.0)),
        },
        "recommended_tools": rec,
    }


def load_image_metrics_map_default(seq: bool) -> Dict[int, Dict[int, Dict[str, Any]]]:
    return load_image_metrics_map(Path("/Users/joshinshaji/dev/GEPA-CATC/gepa_logs/image_diagnostics"))


def _compute_importance_vector(current_level: int, k: int) -> List[float]:
    """Compute a cosine-centered importance vector over k levels.

    Matches CATP-LLM's smoothing in TokenEncoder by assigning higher weight to
    the current level and smoothly decaying weights to neighbors using a cosine
    curve centered at ``current_level``.

    For level index j in [0, k-1], define normalized distance
        d(j) = |j - current_level| / max(1, (k - 1)).
    The unnormalized weight is
        w(j) = cos( d(j) * pi/2 ).
    We then normalize so that sum_j v(j) = 1.
    """
    if k <= 0:
        return []
    if k == 1:
        return [1.0]
    # Clamp current_level to valid range for safety
    cl = max(0, min(current_level, k - 1))
    denom = float(k - 1)
    weights = [math.cos((abs(j - cl) / denom) * (math.pi / 2.0)) for j in range(k)]
    s = sum(weights)
    if s <= 0:
        return [0.0 for _ in range(k)]
    return [w / s for w in weights]


def _build_augmented_tool_catalog_json(input_attributes: Dict, current_size_level: Optional[int] = None) -> str:
    """Build a tool catalog JSON augmented with cost-aware context features.

    - Categorizes input size into levels using GlobalDataConfig.
    - Includes per-tool cost attribute vector c(t_i) across k levels (from GlobalToolConfig.tool_prices).
    - Adds importance vector v computed from the current input level.
    - Provides per-tool cost-aware features as element-wise product c(t_i) * v and a weighted sum.
    
    If ``current_size_level`` is provided, it will be used as the current input
    size level (preferred, e.g., when computed via ``determine_sample_size``). If
    not provided, the level will be inferred from ``input_attributes`` using
    ``determine_input_level``.
    """

    # Current input size level (l)
    level = current_size_level if current_size_level is not None else determine_input_level(input_attributes)
    k = len(next(iter(GlobalToolConfig.tool_prices.values()))) if GlobalToolConfig.tool_prices else 0
    importance = _compute_importance_vector(level if level is not None else 0, k)

    tools = []
    for name, prices in GlobalToolConfig.tool_prices.items():
        io = GlobalToolConfig.tool_io_dict.get(name, ["unknown", "unknown"]) 
        # cost-aware features
        if len(importance) == len(prices):
            cost_features = [p * w for p, w in zip(prices, importance)]
            weighted_cost = sum(cost_features)
        else:
            cost_features = list(prices)
            weighted_cost = sum(prices) / len(prices) if prices else 0.0

        tools.append({
            "name": name,
            "description": GlobalToolConfig.tool_descriptions.get(name, ""),
            "input_type": io[0],
            "output_type": io[1],
            # augmented cost-aware context
            "tool_cost": weighted_cost,
        })
        
    return json.dumps(tools)


def _build_tool_catalog_json() -> str:
    """
    Legacy plain tool catalog without context augmentation.
    DEPRECATED: Use _build_augmented_tool_catalog_json instead.
    """
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


def _get_input_attributes(task_id: int, sample_id: int) -> Dict:
    """Fetch input attributes compatible with CATP-LLM for the given task/sample.

    Falls back to empty attributes if the dataset files are unavailable.
    """
    try:
        _, sample_info = get_task_and_sample_info(task_id, sample_id, data_path=GlobalPathConfig.data_path)
        # sample_info already follows the desired schema
        return dict(sample_info)
    except Exception as e:
        print(e)
        # Best-effort fallback: unknown input
        return {"has_image": False, "image_size": None, "has_text": False, "text_length": None}


def _load_task_descriptions(repo_root: Path) -> List[str]:
    desc_path = repo_root / "catp_base/dataset/task_descriptions.txt"
    if not desc_path.exists():
        return []
    with desc_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _task_query_for(task_id: int, desc: List[str]) -> Optional[str]:
    if 0 <= task_id < len(desc):
        return desc[task_id]
    return None


def build_valid_plans_examples(
    dataset: CATPDataset,
    train_size: int,
    test_size: int,
    seed: int = 0,
    image_metrics_map: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    examples: List[dspy.Example] = []
    # Also collect per-task buckets as we go so we can split by task IDs
    examples_by_task_build: Dict[int, List[dspy.Example]] = {}
    # Lazily load task descriptions only if needed (fallback path)
    task_descriptions = None  # type: Optional[List[str]]

    for task_id, task_samples in dataset.tasks().items():
        for sample_id, variants in task_samples.items():
            if len(variants) == 0:
                continue
            gold_idx = max(range(len(variants)), key=lambda i: (variants[i].qop or 0.0))
            task_query: Optional[str] = None
            plan_variants = []
            input_attributes = _get_input_attributes(task_id, sample_id)
            # If possible, compute the discrete size level used by CATP-LLM
            try:
                size_level = determine_sample_size(input_attributes, task_id, sample_id, data_path=GlobalPathConfig.data_path)
            except Exception:
                size_level = determine_input_level(input_attributes)
            augmented_tool_catalog_json = _build_augmented_tool_catalog_json({**input_attributes}, current_size_level=size_level)
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
                # Only load original descriptions if dataset lacks task_query
                if task_descriptions is None:
                    repo_root = Path(__file__).parent.parent
                    task_descriptions = _load_task_descriptions(repo_root)
                task_query = _task_query_for(task_id, task_descriptions) or ""
            plan_variants_json = json.dumps(plan_variants)
            gold_plan = variants[gold_idx].plan
            gold_qop = variants[gold_idx].qop if variants[gold_idx].qop is not None else 0.0
            gold_plan_json = json.dumps(gold_plan)
            enriched_attrs = dict(input_attributes)
            if image_metrics_map:
                try:
                    m_src = image_metrics_map.get(int(task_id), {}).get(int(sample_id))  # type: ignore[arg-type]
                except Exception:
                    m_src = None
                if isinstance(m_src, dict):
                    mm = _select_relevant_image_metrics(m_src)
                    t = _derive_targets(task_query or "", mm)
                    enriched_attrs["image_metrics"] = mm
                    enriched_attrs["restoration_targets"] = t

            ex = dspy.Example({
                "task_id": task_id,
                "sample_id": sample_id,
                "task_query": task_query,
                "tool_catalog_json_with_description": f"These are all the tools available to you which also include the cost of each tool for the current image size:\n\n ```{augmented_tool_catalog_json}```",
                "tool_catalog_json": augmented_tool_catalog_json,
                "plan_variants_json": plan_variants_json,
                "gold_plan_json": gold_plan_json,
                "gold_qop": gold_qop,
                "input_attributes_json": json.dumps(enriched_attrs),
                "current_size_level": size_level,
            }).with_inputs("task_query", "tool_catalog_json_with_description", "input_attributes_json")
            examples.append(ex)
            examples_by_task_build.setdefault(int(task_id), []).append(ex)

    rng = random.Random(seed)

    # Select whole tasks for the test set to avoid task ID overlap with train
    all_task_ids = list(examples_by_task_build.keys())
    rng.shuffle(all_task_ids)
    selected_test_task_ids: List[int] = []
    accumulated = 0
    for tid in all_task_ids:
        if test_size <= 0:
            break
        bucket = examples_by_task_build[tid]
        if accumulated < test_size:
            selected_test_task_ids.append(tid)
            accumulated += len(bucket)
        else:
            break

    # Build test set from selected task IDs, shuffle within, and cap to test_size
    test_set: List[dspy.Example] = []
    for tid in selected_test_task_ids:
        test_set.extend(examples_by_task_build[tid])
    rng.shuffle(test_set)
    if test_size > 0:
        test_set = test_set[:test_size]

    # Remaining examples (from non-test tasks) will form train/val pools
    remaining_task_ids = [tid for tid in all_task_ids if tid not in selected_test_task_ids]
    examples_by_task: Dict[int, List[dspy.Example]] = {tid: list(examples_by_task_build[tid]) for tid in remaining_task_ids}

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
