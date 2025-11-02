"""
Generate tool plans using image-quality metrics as additional context.

Usage example:

    python catp_experiments/get_tool_plans_image_metrics.py \
        --dataset-path catp-llm/dataset \
        --image-metrics catp_experiments/image_metrics_data/seq_data_with_image_metrics/valid_plans_all.json \
        --prompt-path catp_experiments/prompts/gepa_prompt_nonseq_image_metrics.txt \
        --provider together \
        --default-task-set seq \
        --output-path catp_experiments/output_jsons/gepa-metrics/qwen2.5_7B_seq_img_metrics.json

Environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, TOGETHER_API_KEY (depending on provider)
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# LLM Provider Abstractions
# ---------------------------------------------------------------------------

load_dotenv()

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def infer(self, prompt: str) -> str:
        """Send prompt to the LLM and return text response."""


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def infer(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def infer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content if response.choices else ""


class TogetherProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        import together

        self.client = together.Client(api_key=api_key)
        self.model = model

    def infer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content if response.choices else ""


def load_provider(name: str, model: str | None) -> BaseLLMProvider:
    name = name.lower()
    if name == "anthropic":
        return AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model=model or "claude-sonnet-4-20250514")
    if name == "openai":
        return OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], model=model or "gpt-4o-mini")
    if name == "together":
        return TogetherProvider(api_key=os.environ["TOGETHER_API_KEY"], model=model or "Qwen/Qwen2.5-7B-Instruct-Turbo")
    raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Prompt Helpers
# ---------------------------------------------------------------------------


def get_tool_prices(image_size: Tuple[int, int]) -> Dict[str, float]:
    reference_areas = [490 * 402, 582 * 578, 954 * 806, 1921 * 2624]
    tool_prices = {
        "image_colorization": [0.02064448408770396, 0.030365631874341062, 0.03283418516591483, 0.20662706942016817],
        "image_captioning": [0.07681041316118693, 0.0789306347162341, 0.0856651295420851, 0.15344758910825465],
        "image_classification": [0.0018386416371498659, 0.002039574896674113, 0.001632744822041972, 0.003468827901294199],
        "image_deblurring": [0.044596890616843535, 0.0763150490990877, 0.1770620811739597, 1.1910876856898132],
        "image_denoising": [0.03310938232453873, 0.05553857351136441, 0.126780926718445, 0.8438447906436202],
        "image_super_resolution": [0.12710071314082524, 0.17339502868541215, 0.3204404321613257, 1.8004475979612815],
        "machine_translation": [0.0514501296757946, 0.07495833137293198, 0.17899568650893222, 0.1265572219738673],
        "object_detection": [0.008221186114819342, 0.007555937255849043, 0.007457512004008459, 0.010285611128207868],
    }

    area = image_size[0] * image_size[1]
    sims = np.array([min(area, ref) / max(area, ref) for ref in reference_areas])
    sims /= sims.sum()
    return {tool: float(np.dot(prices, sims)) for tool, prices in tool_prices.items()}


def render_prompt(template: str, task_query: str, tool_prices: Dict[str, float], input_attributes: Dict[str, Any]) -> str:
    return (
        template.replace("{task_query}", task_query)
        .replace("{tool_prices}", json.dumps(tool_prices, indent=2))
        .replace("{input_attributes}", json.dumps(input_attributes, indent=2))
    )


# ---------------------------------------------------------------------------
# Image metrics utilities
# ---------------------------------------------------------------------------


def load_image_metrics_map(path: str | Path | None = None) -> dict[int, dict[int, dict[str, Any]]]:
    default_dir = Path("/Users/joshinshaji/dev/GEPA-CATC/gepa_logs/image_diagnostics")
    p = Path(path) if path else default_dir
    if not p.exists():
        return {}
    if p.is_dir():
        out: dict[int, dict[int, dict[str, Any]]] = {}
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
    out: dict[int, dict[int, dict[str, Any]]] = {}
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

def _select_relevant_image_metrics(m: dict[str, Any]) -> dict[str, Any]:
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

def _derive_targets(task_query: str, m: dict[str, Any]) -> dict[str, Any]:
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
    }


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


DEFAULT_TASKS = {
    "seq": [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81],
    "nonseq": [200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221],
}


def iter_image_indices() -> Iterable[int]:
    for idx in range(100):
        yield idx


def load_task_descriptions(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


def generate_tool_plans(
    tasks: Sequence[int],
    provider: BaseLLMProvider,
    *,
    dataset_root: Path,
    prompt_template: str,
    image_metrics_map: Dict[int, Dict[int, Dict[str, Any]]],
    task_descriptions: List[str],
    output_path: Path,
) -> None:
    results: Dict[str, Dict[str, Dict[str, str]]] = {}

    for task in tasks:
        task_results: Dict[str, Dict[str, str]] = {}
        task_dir = dataset_root / str(task) / "inputs" / "images"

        for image_idx in iter_image_indices():
            image_path = task_dir / f"{image_idx}.jpg"
            if not image_path.exists():
                break

            print(f"Task {task} | Image {image_idx}")
            with Image.open(image_path) as img:
                image_size = img.size

            task_query = task_descriptions[int(task)]
            metrics = image_metrics_map.get(int(task), {}).get(int(image_idx)) or {}
            metrics_selected = _select_relevant_image_metrics(metrics)
            targets = _derive_targets(task_query, metrics_selected)

            input_attributes = {
                "image_metrics": metrics_selected,
                "restoration_targets": targets,
            }

            tool_prices = get_tool_prices(image_size)
            prompt = render_prompt(prompt_template, task_query, input_attributes, tool_prices)
            plan = provider.infer(prompt)

            task_results[str(image_idx)] = {"plan": plan}
            print(plan)

        results[str(task)] = task_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} tasks to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CATP tool plans with image metrics context.")
    parser.add_argument("--dataset-path", default="catp-llm/dataset", help="Path to CATP dataset root.")
    parser.add_argument(
        "--image-metrics",
        default="catp_experiments/image_metrics_data/seq_data_with_image_metrics/valid_plans_all.json",
        help="JSON file containing per-image metric annotations.",
    )
    parser.add_argument(
        "--prompt-path",
        default="catp_experiments/prompts/gepa_prompt_nonseq_image_metrics.txt",
        help="Prompt template used for inference.",
    )
    parser.add_argument("--provider", choices=["anthropic", "openai", "together"], default="together", help="LLM provider to query.")
    parser.add_argument("--provider-model", help="Override default provider model name.")
    parser.add_argument("--default-task-set", choices=list(DEFAULT_TASKS.keys()), default="seq", help="Task subset to evaluate.")
    parser.add_argument("--output-path", required=True, help="File path to write resulting JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_path).expanduser()
    prompt_template = Path(args.prompt_path).expanduser().read_text(encoding="utf-8")
    image_metrics_map = load_image_metrics_map(Path(args.image_metrics).expanduser())
    task_descriptions = load_task_descriptions(dataset_root / "task_descriptions.txt")
    tasks = DEFAULT_TASKS[args.default_task_set]

    provider = load_provider(args.provider, args.provider_model)

    generate_tool_plans(
        tasks,
        provider,
        dataset_root=dataset_root,
        prompt_template=prompt_template,
        image_metrics_map=image_metrics_map,
        task_descriptions=task_descriptions,
        output_path=Path(args.output_path).expanduser(),
    )


if __name__ == "__main__":
    main()
