"""
Step 1 of the SFT pipeline: dataset ingestion and prompt/response formatting.
<<<<<<< HEAD
=======

This module is intentionally focused on data preparation only. It provides
helpers that:
  * load the GRPO-style JSON with task → images → plans
  * render prompts using the same cost-aware template as training/inference
  * normalize the gold plan so it becomes a clean text target
  * split tasks into train/eval partitions and surface HuggingFace datasets

Later steps (model loading, optimization, CLI wiring) will be layered on top,
but for now we keep the public surface limited to data utilities so they can
be unit-tested in isolation.
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
"""

from __future__ import annotations

import argparse
<<<<<<< HEAD
import inspect
=======
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

<<<<<<< HEAD
import numpy as np
=======
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)
<<<<<<< HEAD
try:
    from trl import SFTConfig, SFTTrainer
except ImportError:  # pragma: no cover - older TRL versions
    from trl import SFTTrainer

    SFTConfig = None


SFT_TRAINER_SUPPORTS_DATASET_FIELD = "dataset_text_field" in inspect.signature(SFTTrainer.__init__).parameters
=======
from trl import SFTTrainer
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """Structured representation of one prompt/response pair."""

    sample_id: str
    task_id: str
    image_id: str
    prompt: str
    response: str
    task_query: str
    image_width: int
    image_height: int
    qop: float
    task_score: float

    def to_dict(self) -> Dict:
        """HF Datasets expects regular dicts; convert lazily."""
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "image_id": self.image_id,
            "prompt": self.prompt,
            "response": self.response,
            "task_query": self.task_query,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "qop": self.qop,
            "task_score": self.task_score,
        }


# ---------------------------------------------------------------------------
# Prompt helpers (reused later by training/inference)
# ---------------------------------------------------------------------------


def compute_tool_prices(image_size: Tuple[int, int]) -> Dict[str, float]:
    """Blend discrete tool prices to the current image size via area similarity."""
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
    similarities = [min(area, ref) / max(area, ref) for ref in reference_areas]
    denom = sum(similarities) or 1.0
    weights = [value / denom for value in similarities]

    blended = {}
    for tool, prices in tool_prices.items():
        blended[tool] = float(sum(price * weight for price, weight in zip(prices, weights)))
    return blended


def load_prompt_template(path: Path) -> str:
    """Return the raw template text."""
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, task_query: str, image_size: Tuple[int, int], tool_prices: Dict[str, float], sample_id: str) -> str:
    """Fill the prompt template with per-sample metadata."""
    prefix = f"### SAMPLE_ID: {sample_id}\n"
    prompt = template.replace("{task_query}", task_query)
    prompt = prompt.replace("{input_size}", str(list(image_size)))
    prompt = prompt.replace("{tool_prices}", json.dumps(tool_prices, indent=2))
    return prefix + prompt


# ---------------------------------------------------------------------------
# Plan normalization logic
# ---------------------------------------------------------------------------


def try_parse_plan(plan_text: str) -> Optional[List]:
    """Attempt to parse the alternating plan format, tolerating single quotes."""
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        pass

    try:
        import ast

        parsed = ast.literal_eval(plan_text)
    except (ValueError, SyntaxError):
        return None

    return parsed if isinstance(parsed, list) else None


def normalize_plan_string(plan_text: str) -> str:
    """
    Convert a plan into a deterministic repr so the model always sees the same target.

    SFT works best when the supervision string is stable (no extra whitespace, double
    quotes vs single quotes, etc.), so we attempt to parse the plan and then emit a
    canonical repr. If parsing fails we fall back to the raw text to avoid data loss.
    """
    parsed = try_parse_plan(plan_text)
    if parsed is None:
        return plan_text.strip()
    return repr(parsed)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def split_tasks(task_ids: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Task-level split so all samples from a task stay within the same partition."""
    shuffled = task_ids[:]
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []
    split_idx = max(1, int(len(shuffled) * train_ratio))
    if len(shuffled) > 1:
        split_idx = min(split_idx, len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def _iter_sample_records(
    dataset_payload: Dict[str, Dict],
    prompt_template: str,
) -> Iterable[SampleRecord]:
    """Yield structured records that combine prompts with normalized responses."""
    for task_id, task_payload in dataset_payload.items():
        task_query = task_payload["task_query"]
        for image_id, image_payload in task_payload["images"].items():
            image_size = tuple(image_payload["image_size"])
            sample_id = f"{task_id}:{image_id}"
            prompt = render_prompt(
                template=prompt_template,
                task_query=task_query,
                image_size=image_size,
                tool_prices=compute_tool_prices(image_size),
                sample_id=sample_id,
            )
            gold_plan = image_payload["gold_plan"]

            yield SampleRecord(
                sample_id=sample_id,
                task_id=task_id,
                image_id=image_id,
                prompt=prompt,
                response=normalize_plan_string(gold_plan["plan"]),
                task_query=task_query,
                image_width=image_size[0],
                image_height=image_size[1],
                qop=float(gold_plan.get("qop", 0.0)),
                task_score=float(gold_plan.get("task_score", 0.0)),
            )


def build_sft_datasets(
    dataset_path: Path,
    prompt_template_path: Path,
    train_ratio: float,
    split_seed: int,
) -> DatasetDict:
    """
    Materialize HuggingFace Dataset splits for supervised fine-tuning.

    Args:
        dataset_path: Path to grpo_dataset_(seq|nonseq).json.
        prompt_template_path: Prompt template with {task_query}/{input_size}/{tool_prices}.
        train_ratio: Fraction of tasks to assign to the train partition.
        split_seed: RNG seed for reproducible task-level shuffling.

    Returns:
        DatasetDict with `train` and optionally `eval` splits.
    """
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    prompt_template = load_prompt_template(prompt_template_path)

    task_ids = sorted(dataset_payload.keys(), key=lambda value: int(value))
    train_tasks, eval_tasks = split_tasks(task_ids, train_ratio, split_seed)
    train_set = set(train_tasks)

    split_buckets = {"train": [], "eval": []}
    for record in _iter_sample_records(dataset_payload, prompt_template):
        split_key = "train" if record.task_id in train_set else "eval"
        split_buckets[split_key].append(record.to_dict())

    dataset_dict = {}
    for split_name, records in split_buckets.items():
        if records:
            dataset_dict[split_name] = Dataset.from_list(records)

    if "train" not in dataset_dict:
        raise ValueError("Training split is empty; check train_ratio or dataset integrity.")

    return DatasetDict(dataset_dict)


# ---------------------------------------------------------------------------
# Step 2: tokenizer/model utilities (to be reused by the training harness)
# ---------------------------------------------------------------------------


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool) -> Optional[BitsAndBytesConfig]:
    """Return a bitsandbytes quantization config when requested."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def resolve_torch_dtype(choice: Optional[str]) -> Optional[torch.dtype]:
    """Map CLI-friendly dtype strings to actual torch dtypes."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get((choice or "").lower())


def load_tokenizer(model_name: str, *, trust_remote_code: bool = False):
    """
    Initialize the tokenizer and guarantee padding tokens are available.

    Most chat models reuse eos as the pad token when it is undefined; we follow
    that convention so batching works out of the box in later training stages.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_name: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    quant_config: Optional[BitsAndBytesConfig] = None,
    device_map: Optional[str | Dict] = None,
    trust_remote_code: bool = False,
):
    """
    Load the base AutoModelForCausalLM with the desired precision/quantization.

    The returned model can later be wrapped by PEFT/LoRA or fed directly into
    SFTTrainer once optimizer settings are in place.
    """
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def maybe_apply_lora(
    model,
    *,
    use_lora: bool,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[str] = None,
    loaded_in_kbit: bool = False,
):
    """
    Optionally wrap the backbone with LoRA adapters.

    Args:
        model: Base AutoModelForCausalLM.
        use_lora: Whether to inject adapters.
        lora_*: Standard LoRA hyperparameters.
        lora_target_modules: Comma-separated list; defaults match common QKV/MLP proj names.
        loaded_in_kbit: If the model was quantized (4/8-bit), we run PEFT's
                        prepare_model_for_kbit_training before adding adapters.
    """
    if not use_lora:
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    target_modules = (
        lora_target_modules.split(",")
        if lora_target_modules
        else [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    if loaded_in_kbit:
        model = prepare_model_for_kbit_training(model)

    return get_peft_model(model, lora_config)


# ---------------------------------------------------------------------------
# Step 3: minimal training orchestration helpers
# ---------------------------------------------------------------------------


def prepare_datasets_with_text_column(dataset_dict: DatasetDict, tokenizer, *, num_proc: Optional[int] = None) -> DatasetDict:
    """
    Convert prompt/response pairs into a single `text` field for SFTTrainer.

    We keep this logic close to the training harness, because later we may want
    to add packing/concatenation logic; for now each record is a simple chat.
    """

    has_chat_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    def _format(record: Dict) -> Dict:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
        if has_chat_template:
            record["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            record["text"] = record["prompt"].rstrip() + "\n\n### Response:\n" + record["response"]
<<<<<<< HEAD
        record.pop("prompt", None)
        record.pop("response", None)
=======
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
        return record

    formatted = DatasetDict()
    for split_name, dataset in dataset_dict.items():
        formatted[split_name] = dataset.map(_format, desc=f"Formatting {split_name} split", num_proc=num_proc)
    return formatted


def build_training_components(
    dataset_json: Path,
    prompt_template: Path,
    train_ratio: float,
    split_seed: int,
    base_model: str,
    *,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    dataset_num_proc: Optional[int] = None,
    torch_dtype: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = False,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[str] = None,
) -> Tuple[DatasetDict, AutoTokenizer, AutoModelForCausalLM]:
    """Bundle dataset/tokenizer/model creation for the training script."""

    dataset_dict = build_sft_datasets(dataset_json, prompt_template, train_ratio, split_seed)

    if max_train_samples and max_train_samples < len(dataset_dict["train"]):
        dataset_dict["train"] = dataset_dict["train"].select(range(max_train_samples))
    if "eval" in dataset_dict and max_eval_samples and max_eval_samples < len(dataset_dict["eval"]):
        dataset_dict["eval"] = dataset_dict["eval"].select(range(max_eval_samples))

    quant_config = build_quantization_config(load_in_4bit, load_in_8bit)
    dtype = resolve_torch_dtype(torch_dtype)

    tokenizer = load_tokenizer(base_model, trust_remote_code=trust_remote_code)
    datasets_with_text = prepare_datasets_with_text_column(dataset_dict, tokenizer, num_proc=dataset_num_proc)

    model = load_causal_lm(
        base_model,
        torch_dtype=dtype,
        quant_config=quant_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    model = maybe_apply_lora(
        model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        loaded_in_kbit=bool(load_in_4bit or load_in_8bit),
    )

    return datasets_with_text, tokenizer, model


<<<<<<< HEAD
def compute_token_accuracy(eval_pred) -> Dict[str, float]:
    """Compute masked token-level accuracy for evaluation logging."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if predictions is None or labels is None:
        return {}

    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.ndim == labels.ndim + 1:
        predictions = predictions.argmax(axis=-1)

    mask = labels != -100
    total = mask.sum()
    if total == 0:
        return {"eval_accuracy": 0.0}

    accuracy = (predictions[mask] == labels[mask]).mean()
    return {"eval_accuracy": float(accuracy)}


=======
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
def run_sft_training(
    datasets: DatasetDict,
    tokenizer,
    model,
    *,
    output_dir: str,
    num_train_epochs: float,
    train_batch_size: int,
    eval_batch_size: Optional[int],
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    save_strategy: str,
    max_grad_norm: float,
    lr_scheduler_type: str,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
    report_to: Optional[List[str]] = None,
    run_name: Optional[str] = None,
    max_seq_length: int = 4096,
    packing: bool = False,
):
    """Spin up TRL's SFTTrainer with the provided hyperparameters."""

    train_dataset = datasets["train"]
    eval_dataset = datasets.get("eval")

<<<<<<< HEAD
    logging.info("Train dataset size: %d", len(train_dataset))
    logging.info("Eval dataset size: %d", len(eval_dataset) if eval_dataset is not None else 0)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    evaluation_strategy = "steps" if eval_dataset is not None else "no"

    training_kwargs = dict(
=======
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    evaluation_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size or train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
<<<<<<< HEAD
=======
        evaluation_strategy=evaluation_strategy,
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
        logging_strategy="steps",
        report_to=report_to or [],
        run_name=run_name,
        ddp_find_unused_parameters=False,
    )

<<<<<<< HEAD
    eval_field_name = (
        "evaluation_strategy" if "evaluation_strategy" in TrainingArguments.__dataclass_fields__ else "eval_strategy"
    )
    training_kwargs[eval_field_name] = evaluation_strategy

    if not SFT_TRAINER_SUPPORTS_DATASET_FIELD and SFTConfig is not None:
        sft_kwargs = dict(
            dataset_text_field="text",
            max_length=max_seq_length,
            packing=packing,
        )
        training_args = SFTConfig(**{**training_kwargs, **sft_kwargs})
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_token_accuracy,
        )
    else:
        training_args = TrainingArguments(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=packing,
            compute_metrics=compute_token_accuracy,
        )
=======
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=packing,
    )
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)

    trainer.train()
    if eval_dataset is not None:
        trainer.evaluate()
    trainer.save_state()
    trainer.save_model()


# ---------------------------------------------------------------------------
# Step 4: CLI wiring / end-to-end entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training harness for CATP plans")
<<<<<<< HEAD
    parser.add_argument("--dataset-json")
    parser.add_argument("--prompt-template")
=======
    parser.add_argument("--dataset-json", default="catp_grpo/grpo_dataset_seq.json")
    parser.add_argument("--prompt-template", default="catp_experiments/prompts/gepa_prompt_seq.txt")
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--dataset-num-proc", type=int)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="sft-qwen2.5-7b")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
<<<<<<< HEAD
    parser.add_argument("--eval-batch-size", type=int, default=1)
=======
    parser.add_argument("--eval-batch-size", type=int)
>>>>>>> dd3e92a (Added argument parsers and cleaned up the code)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules")
    parser.add_argument("--report-to")
    parser.add_argument("--run-name", default="catp-sft")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    set_seed(args.seed)

    datasets, tokenizer, model = build_training_components(
        dataset_json=Path(args.dataset_json),
        prompt_template=Path(args.prompt_template),
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        base_model=args.base_model,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        dataset_num_proc=args.dataset_num_proc,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    report_to = args.report_to.split(",") if args.report_to else []

    run_sft_training(
        datasets,
        tokenizer,
        model,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=report_to,
        run_name=args.run_name,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )


if __name__ == "__main__":
    main()
