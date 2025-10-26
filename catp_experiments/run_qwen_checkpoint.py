import os
import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from PIL import Image
import numpy as np


# ----------------------------
# Utility Functions - Change these when running on Pod
# ----------------------------

repo_root = Path(__file__).resolve().parents[1]
dataset_root = repo_root / "catp-llm" / "dataset"
task_descriptions_path = dataset_root / "task_descriptions.txt"

MODES: Dict[str, Dict[str, str]] = {
    "seq": {
        "prompt_path": "/workspace/GEPA-CATC/catp_experiments/prompts/gepa_prompt_seq.txt",
        "output_json": "/workspace/GEPA-CATC/catp_experiments/output_jsons/sft-gepa/sft_qwen7B_seq_gepa_checkpoint.json",
    },
    "nonseq": {
        "prompt_path": "/workspace/GEPA-CATC/catp_experiments/prompts/prompt_nonseq.txt",
        "output_json": "/workspace/GEPA-CATC/catp_experiments/output_jsons/sft-base/sft_qwen7B_nonseq_checkpoint.json",
    },
}

# Select the evaluation mode here ("seq" or "nonseq")
evaluation_mode = "nonseq"

prompt_path = repo_root / MODES[evaluation_mode]["prompt_path"]
output_path = repo_root / MODES[evaluation_mode]["output_json"]

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
max_new_tokens = 256
temperature = 0.7
top_p = 0.9


# ----------------------------
# Plan Provider Abstraction
# ----------------------------


def format_prompt_with_chat_template(prompt: str, tokenizer) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


class BasePlanProvider(ABC):
    """Abstract base class for plan generation providers."""

    @abstractmethod
    def infer(self, prompt: str) -> str:
        """Send prompt to the model and return text response."""
        raise NotImplementedError


def try_parse_plan(plan_text: str):
    cleaned = (plan_text or "").strip()
    if not cleaned:
        return None
    try:
        parsed = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or len(parsed) % 2 != 0:
        return None
    for idx in range(0, len(parsed), 2):
        tool = parsed[idx]
        deps = parsed[idx + 1] if idx + 1 < len(parsed) else []
        if not isinstance(tool, str) or not isinstance(deps, list):
            return None
        if not all(isinstance(dep, str) for dep in deps):
            return None
    return parsed


def normalize_plan_string(plan_text: str) -> str:
    parsed = try_parse_plan(plan_text)
    if parsed is None:
        return (plan_text or "").strip()
    return repr(parsed)


class LoRAProvider(BasePlanProvider):
    def __init__(
        self,
        checkpoint_directory: Path,
        model_name: str = base_model_name,
        *,
        temperature: float = temperature,
        top_p: float = top_p,
        max_new_tokens: int = max_new_tokens,
    ):
        self.checkpoint_directory = Path(checkpoint_directory).expanduser().resolve()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        adapter_dir = self.checkpoint_directory
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Checkpoint {adapter_dir} missing adapter_config.json")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def infer(self, prompt: str) -> str:
        formatted_prompt = format_prompt_with_chat_template(prompt, self.tokenizer)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def get_tool_prices(image_size: Tuple[int, int]):
    image_sizes = [490 * 402, 582 * 578, 954 * 806, 1921 * 2624]

    tool_prices = {
        "image_colorization": [0.02064448408770396, 0.030365631874341062, 0.03283418516591483, 0.20662706942016817],
        "image_captioning": [0.07681041316118693, 0.0789306347162341, 0.0856651295420851, 0.15344758910825465],
        "image_classification": [0.0018386416371498659, 0.002039574896674113, 0.001632744822041972, 0.003468827901294199],
        "image_deblurring": [0.044596890616843535, 0.0763150490990877, 0.1770620811739597, 1.1910876856898132],
        "image_denoising": [0.03310938232453873, 0.05553857351136441, 0.126780926718445, 0.8438447906436202],
        "image_super_resolution": [0.12710071314082524, 0.17339502868541215, 0.3204404321613257, 1.8004475979612815],
        "object_detection": [0.008221186114819342, 0.007555937255849043, 0.007457512004008459, 0.010285611128207868],
        "machine_translation": [0.0514501296757946, 0.07495833137293198, 0.17899568650893222, 0.1265572219738673],
    }

    input_area = image_size[0] * image_size[1]
    sims = np.array([min(input_area, ref) / max(input_area, ref) for ref in image_sizes])
    sims /= sims.sum()

    custom_tool_prices = {
        tool: float(np.dot(prices, sims))
        for tool, prices in tool_prices.items()
    }

    return custom_tool_prices


def load_prompt(task: str, input_size: Tuple[int, int], custom_tool_prices: Dict[str, float], template_file: Path = prompt_path) -> str:
    with open(template_file, "r", encoding="utf-8") as f:
        template = f.read()
    return (
        template
        .replace("{task_query}", task)
        .replace("{input_size}", str(list(input_size)))
        .replace("{tool_prices}", json.dumps(custom_tool_prices, indent=2))
    )


def read_task_by_index(index: int) -> str:
    with open(task_descriptions_path, "r", encoding="utf-8") as f:
        tasks = [line.rstrip("\n") for line in f]
    return tasks[index]


def output_json(
    task_list: List[int],
    file_name: Path,
    provider: BasePlanProvider,
) -> None:
    results: Dict[str, Dict] = {}
    for task in task_list:
        task_results: Dict[str, Dict] = {}
        for image_index in range(100):
            print("Task: ", task, "Image: ", image_index)
            img_path = dataset_root / str(task) / "inputs" / "images" / f"{image_index}.jpg"
            if not img_path.exists():
                continue
            with Image.open(img_path) as img:
                image_size = img.size

            task_str = read_task_by_index(int(task))
            custom_tool_prices = get_tool_prices(image_size)

            prompt = load_prompt(task_str, image_size, custom_tool_prices)
            raw_plan = provider.infer(prompt)
            normalized_plan = normalize_plan_string(raw_plan)
            if try_parse_plan(normalized_plan) is None and raw_plan.strip():
                print("WARNING: Generated plan could not be parsed as a valid plan.")
            task_results[str(image_index)] = {"plan": normalized_plan}
            print(normalized_plan)
        results[str(task)] = task_results

    file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    default_test_seq_tasks = [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81]
    default_test_nonseq_tasks = [200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221]

    checkpoint_path = Path("/workspace/GEPA-CATC/catp_sft/runs/sft-qwen-nonseq/checkpoint-153")

    if evaluation_mode == "seq":
        selected_tasks = default_test_seq_tasks
    else:
        selected_tasks = default_test_nonseq_tasks

    provider = LoRAProvider(checkpoint_path)
    output_json(selected_tasks, output_path, provider)
