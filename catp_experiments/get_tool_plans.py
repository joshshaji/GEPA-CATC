"""
Generate tool plans from an LLM for a set of CATP tasks.

Usage example:

    python catp_experiments/get_tool_plans.py \
        --dataset-path /Users/mdamarap/GEPA-CATC/catp-llm/dataset \
        --prompt-path catp_experiments/prompts/gepa_prompt_nonseq.txt \
        --provider together \
        --output-path catp_experiments/output_jsons/qwen2.5_7B_nonseq_gepa_1.json \
        --default-task-set nonseq

Environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, TOGETHER_API_KEY (depending on provider)
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# LLM Provider abstractions
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def infer(self, prompt: str) -> str:
        """Send prompt to the LLM and return text response."""
        raise NotImplementedError


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
    """Instantiate an LLM provider using environment credentials."""
    name = name.lower()
    if name == "anthropic":
        api_key = os.environ["ANTHROPIC_API_KEY"]
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    if name == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")
    if name == "together":
        api_key = os.environ["TOGETHER_API_KEY"]
        return TogetherProvider(api_key=api_key, model=model or "Qwen/Qwen2.5-7B-Instruct-Turbo")
    raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------


def get_tool_prices(image_size: Tuple[int, int]) -> Dict[str, float]:
    """Interpolate tool prices to the current image size."""
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

    return {
        tool: float(np.dot(prices, sims))
        for tool, prices in tool_prices.items()
    }


def render_prompt(template: str, task: str, input_size: Tuple[int, int], tool_prices: Dict[str, float]) -> str:
    """Fill the prompt template with per-sample metadata."""
    return (
        template.replace("{task_query}", task)
        .replace("{input_size}", str(input_size))
        .replace("{tool_prices}", json.dumps(tool_prices, indent=2))
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_task_descriptions(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


DEFAULT_TASKS = {
    "seq": [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81],
    "nonseq": [200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221],
}


def iter_image_indices() -> Iterable[int]:
    """Yield image indices to evaluate for each task."""
    for idx in range(100):
        yield idx


# ---------------------------------------------------------------------------
# Core generation entry point
# ---------------------------------------------------------------------------


def generate_tool_plans(
    task_list: Sequence[int],
    provider: BaseLLMProvider,
    *,
    dataset_root: Path,
    prompt_template: str,
    task_descriptions: List[str],
    output_path: Path,
) -> None:
    """Run the LLM over each task/image and persist plan outputs."""
    results: Dict[str, Dict[str, Dict[str, str]]] = {}
    for task in task_list:
        task = int(task)
        task_results: Dict[str, Dict[str, str]] = {}
        task_dir = dataset_root / str(task) / "inputs" / "images"
        task_dir.mkdir(parents=False, exist_ok=True)  # Ensures meaningful error if missing

        for image_idx in iter_image_indices():
            image_path = task_dir / f"{image_idx}.jpg"
            if not image_path.exists():
                # Stop early if the dataset does not contain this many images.
                break

            print(f"Task {task} | Image {image_idx}")
            with Image.open(image_path) as img:
                image_size = img.size

            task_query = task_descriptions[task]
            tool_prices = get_tool_prices(image_size)
            prompt = render_prompt(prompt_template, task_query, image_size, tool_prices)
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
    parser = argparse.ArgumentParser(description="Generate CATP tool plans via an LLM.")
    parser.add_argument("--dataset-path", default="/Users/mdamarap/GEPA-CATC/catp-llm/dataset", help="Path to CATP dataset root.")
    parser.add_argument("--prompt-path", default="catp_experiments/prompts/gepa_prompt_nonseq.txt", help="Prompt template used for inference.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "together"], default="together", help="LLM provider to query.")
    parser.add_argument("--provider-model", help="Override default provider model name.")
    parser.add_argument("--default-task-set", choices=list(DEFAULT_TASKS.keys()), default="seq", help="Which predefined task subset to run (seq vs nonseq).")
    parser.add_argument("--output-path", required=True, help="File path to write resulting JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_path).expanduser()
    prompt_template = Path(args.prompt_path).expanduser().read_text(encoding="utf-8")
    task_descriptions = load_task_descriptions(dataset_root / "task_descriptions.txt")
    tasks = DEFAULT_TASKS[args.default_task_set]

    provider = load_provider(args.provider, args.provider_model)

    generate_tool_plans(
        tasks,
        provider,
        dataset_root=dataset_root,
        prompt_template=prompt_template,
        task_descriptions=task_descriptions,
        output_path=Path(args.output_path).expanduser(),
    )


if __name__ == "__main__":
    main()
