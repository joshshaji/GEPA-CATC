import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


logger = logging.getLogger(__name__)

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
    weight_total = sum(similarities) or 1.0
    weights = [value / weight_total for value in similarities]

    blended = {}
    for tool, prices in tool_prices.items():
        blended[tool] = sum(price * weight for price, weight in zip(prices, weights))
    return blended


def load_prompt_template(path: Path) -> str:
    """Return the raw prompt template text."""
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, task_query: str, image_size: Tuple[int, int], tool_prices: Dict[str, float], sample_id: str) -> str:
    """Fill the prompt template with per-sample details and prepend a sample tag."""
    prefix = f"### SAMPLE_ID: {sample_id}\n"
    prompt = template.replace("{task_query}", task_query)
    prompt = prompt.replace("{input_size}", str(list(image_size)))
    prompt = prompt.replace("{tool_prices}", json.dumps(tool_prices, indent=2))
    return prefix + prompt


def format_prompt_with_chat_template(prompt: str, tokenizer) -> str:
    """Wrap a plain-text prompt with the tokenizer's chat template when available."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to apply chat template: %s", exc)
        return prompt

def try_parse_plan(plan_text: str) -> Optional[List]:
    """Safely parse the string plan into alternating tool/dependency entries."""
    try:
        parsed = ast.literal_eval(plan_text)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, list) or len(parsed) % 2 != 0:
        return None

    validated: List = []
    for idx in range(0, len(parsed), 2):
        tool = parsed[idx]
        deps = parsed[idx + 1] if idx + 1 < len(parsed) else []
        if not isinstance(tool, str) or not isinstance(deps, list):
            return None
        if not all(isinstance(dep, str) for dep in deps):
            return None
        validated.append(tool)
        validated.append(list(deps))
    return validated


def canonicalize_plan(plan_list: List) -> str:
    """Serialize the plan into a deterministic JSON string for hashing."""
    return json.dumps(plan_list, separators=(",", ":"), ensure_ascii=False)


def extract_tools(plan_list: List) -> List[str]:
    """Return the ordered list of tool names from a parsed plan."""
    return [plan_list[idx] for idx in range(0, len(plan_list), 2)]


def build_edge_set(plan_list: List) -> List[Tuple[str, Tuple[str, ...]]]:
    """Represent a plan as (tool, parent tuple) edges to compare flow structure."""
    edges: List[Tuple[str, Tuple[str, ...]]] = []
    for idx in range(0, len(plan_list), 2):
        tool = plan_list[idx]
        deps = tuple(plan_list[idx + 1])
        edges.append((tool, deps))
    return edges


def normalize_qop(qop: float, qop_min: float, qop_max: float) -> float:
    """Scale QoP into [0,1] so different samples are comparable."""
    if qop_max - qop_min < 1e-8:
        return 0.5
    return (qop - qop_min) / (qop_max - qop_min)


def _f1_from_sets(gold: Set[str], pred: Set[str]) -> float:
    """Compute F1 between two sets (0..1)."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def plan_to_dag_struct(plan_list: List) -> Dict[str, Set[str] | Set[Tuple[str, str]]]:
    """Convert a parsed plan (tool, deps) into node and edge sets."""
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    last_tool: Optional[str] = None
    for idx in range(0, len(plan_list), 2):
        tool = plan_list[idx]
        deps = plan_list[idx + 1] if idx + 1 < len(plan_list) else []
        if not isinstance(tool, str) or not isinstance(deps, list):
            continue
        nodes.add(tool)
        for dep in deps:
            dep_s = str(dep)
            if dep_s == "input_of_query":
                continue
            if dep_s == "output_of_previous_tool":
                if last_tool is not None:
                    edges.add((last_tool, tool))
                continue
            edges.add((dep_s, tool))
        last_tool = tool
    return {"nodes": nodes, "edges": edges}


def dag_score_between(
    pred_dag: Dict[str, Set[str] | Set[Tuple[str, str]]],
    gold_dag: Dict[str, Set[str] | Set[Tuple[str, str]]],
    *,
    node_weight: float = 0.5,
    edge_weight: float = 0.5,
) -> float:
    """Return similarity in [0,1] between two DAGs, weighting nodes/edges."""
    weight_sum = max(1e-8, node_weight + edge_weight)
    nw, ew = node_weight / weight_sum, edge_weight / weight_sum
    gold_nodes = set(gold_dag.get("nodes", set()))
    pred_nodes = set(pred_dag.get("nodes", set()))
    gold_edges = set(gold_dag.get("edges", set()))
    pred_edges = set(pred_dag.get("edges", set()))
    node_f1 = _f1_from_sets(gold_nodes, pred_nodes)
    edge_f1 = _f1_from_sets(gold_edges, pred_edges)
    return max(0.0, min(1.0, nw * node_f1 + ew * edge_f1))


def create_plan_record(plan_text: str, qop: float, task_score: float) -> Dict:
    """Pre-compute plan metadata used later for reward shaping."""
    parsed = try_parse_plan(plan_text)
    if parsed is None:
        raise ValueError(f"Invalid plan structure: {plan_text}")
    record = {
        "plan": plan_text,
        "parsed": parsed,
        "canonical": canonicalize_plan(parsed),
        "tools": extract_tools(parsed),
        "edges": [(tool, deps) for tool, deps in build_edge_set(parsed)],
        "qop": qop,
        "task_score": task_score,
        "dag": plan_to_dag_struct(parsed),
    }
    return record


def deduplicate_plan_pool(plans: Iterable[Dict]) -> List[Dict]:
    """Drop duplicate plans, keeping the one with the highest QoP."""
    unique: Dict[str, Dict] = {}
    for plan in plans:
        existing = unique.get(plan["canonical"])
        if existing is None or plan["qop"] > existing["qop"]:
            unique[plan["canonical"]] = plan
    return list(unique.values())



def split_tasks(task_ids: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Task-level split so images from the same task stay in the same partition."""
    shuffled = task_ids[:]
    random.Random(seed).shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    split_idx = min(split_idx, len(shuffled) - 1) if len(shuffled) > 1 else 1
    return shuffled[:split_idx], shuffled[split_idx:]


def prepare_plan_datasets(
    dataset_path: Path,
    prompt_template: Path,
    train_ratio: float,
    seed: int,
) -> Tuple[Dataset, List[Dict], Dict[str, Dict], Dict[str, str]]:
    """Materialize the GRPO dataset, prompts, and per-sample metadata."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    template = load_prompt_template(prompt_template)

    task_ids = sorted(data.keys(), key=lambda value: int(value))
    train_tasks, test_tasks = split_tasks(task_ids, train_ratio, seed)
    train_set = set(train_tasks)

    train_records: List[Dict] = []
    test_samples: List[Dict] = []
    metadata_map: Dict[str, Dict] = {}
    prompt_to_sample: Dict[str, str] = {}

    for task_id, payload in data.items():
        task_query = payload["task_query"]
        for image_id, image_payload in payload["images"].items():
            image_size = tuple(image_payload["image_size"])
            sample_id = f"{task_id}:{image_id}"
            prices = compute_tool_prices(image_size)
            prompt = render_prompt(template, task_query, image_size, prices, sample_id)

            plan_records = [
                create_plan_record(plan_entry["plan"], float(plan_entry.get("qop", 0.0)), float(plan_entry.get("task_score", 0.0)))
                for plan_entry in image_payload["plans"]
            ]
            gold_payload = image_payload["gold_plan"]
            gold_record = create_plan_record(gold_payload["plan"], float(gold_payload.get("qop", 0.0)), float(gold_payload.get("task_score", 0.0)))

            plan_pool = deduplicate_plan_pool(plan_records + [gold_record])
            qops = [plan["qop"] for plan in plan_pool]
            qop_min = min(qops)
            qop_max = max(qops)
            gold_record = next(plan for plan in plan_pool if plan["canonical"] == gold_record["canonical"])

            gold_dag = gold_record["dag"]

            metadata = {
                "sample_id": sample_id,
                "prompt": prompt,
                "task_id": task_id,
                "image_id": image_id,
                "task_query": task_query,
                "image_size": list(image_size),
                "tool_prices": prices,
                "gold_plan": gold_record,
                "plan_pool": plan_pool,
                "gold_dag": gold_dag,
                "qop_min": qop_min,
                "qop_max": qop_max,
            }

            metadata_map[sample_id] = metadata
            prompt_to_sample[prompt] = sample_id

            record = {
                "prompt": prompt,
                "sample_id": sample_id,
                "task_id": task_id,
                "image_id": image_id,
            }

            if task_id in train_set:
                train_records.append(record)
            else:
                test_samples.append({"record": record, "metadata": metadata})

    train_dataset = Dataset.from_list(train_records)
    return train_dataset, test_samples, metadata_map, prompt_to_sample


def score_plan_output(plan_text: str, metadata: Dict) -> Dict[str, object]:
    """Assign a dense reward to a generated plan based on match quality."""
    parsed = try_parse_plan(plan_text.strip())
    if parsed is None:
        return {
            "reward": -1.0,
            "match_type": "invalid",
            "parsed_plan": None,
            "canonical": None,
            "reference": None,
            "dag_score": 0.0,
            "dag_loss": 1.0,
        }

    canonical = canonicalize_plan(parsed)
    gold_record = metadata["gold_plan"]
    plan_lookup = {plan["canonical"]: plan for plan in metadata["plan_pool"]}
    gold_dag = metadata["gold_dag"]
    pred_dag = plan_to_dag_struct(parsed)
    dag_score = dag_score_between(pred_dag, gold_dag)

    tool_costs = metadata.get("tool_prices", {})
    extra_nodes = set(pred_dag["nodes"]) - set(gold_dag["nodes"])
    extra_cost = sum(tool_costs.get(tool, 0.0) for tool in extra_nodes)
    cost_penalty_factor = 1.0 / (1.0 + max(0.0, extra_cost))
    dag_score = max(0.0, min(1.0, dag_score * cost_penalty_factor))
    dag_loss_value = 1.0 - dag_score

    if canonical == gold_record["canonical"]:
        reward = max(0.8, dag_score)
        match_type = "gold"
        reference = gold_record
    elif canonical in plan_lookup:
        ref = plan_lookup[canonical]
        reward = max(0.3, dag_score)
        match_type = "valid"
        reference = ref
    else:
        best = max(metadata["plan_pool"], key=lambda plan: dag_score_between(pred_dag, plan["dag"]))
        similarity = dag_score_between(pred_dag, best["dag"])
        gold_tools = gold_record["tools"]
        length_penalty = abs(len(extract_tools(parsed)) - len(gold_tools)) * 0.05
        reward = dag_score + 0.2 * similarity - length_penalty - 0.2
        reward = max(-1.0, min(reward, 0.8))
        match_type = "partial" if reward > 0 else "invalid"
        reference = best

    return {
        "reward": reward,
        "match_type": match_type,
        "parsed_plan": parsed,
        "canonical": canonical,
        "reference": reference,
        "dag_score": dag_score,
        "dag_loss": dag_loss_value,
    }


def build_plan_reward_function(
    metadata_map: Dict[str, Dict],
    prompt_to_sample: Dict[str, str],
) -> callable:
    """Wrap `score_plan_output` into a TRL-compatible reward function."""
    def reward_fn(completions: Sequence[Sequence[dict]], **kwargs) -> List[float]:
        metadatas = kwargs.get("metadatas") or []
        prompts = kwargs.get("prompts") or []

        sample_ids: List[Optional[str]] = []
        for meta in metadatas:
            if isinstance(meta, dict) and "sample_id" in meta:
                sample_ids.append(meta["sample_id"])

        if len(sample_ids) < len(completions):
            for prompt in prompts:
                sample_id = prompt_to_sample.get(prompt)
                if sample_id is not None:
                    sample_ids.append(sample_id)

        while len(sample_ids) < len(completions):
            sample_ids.append(None)

        rewards: List[float] = []
        for completion, sample_id in zip(completions, sample_ids):
            if isinstance(completion, str):
                text = completion
            elif isinstance(completion, Sequence) and completion:
                first = completion[0]
                if isinstance(first, dict):
                    text = first.get("content", "")
                else:
                    text = str(first)
            else:
                text = ""

            metadata = metadata_map.get(sample_id) if sample_id else None
            if metadata is None:
                logger.warning("Missing metadata for sample %s; assigning -1 reward", sample_id)
                rewards.append(-1.0)
                continue
            normalized_text = normalize_plan_string(text)
            score = score_plan_output(normalized_text, metadata)
            rewards.append(score["reward"])
        return rewards

    return reward_fn


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool) -> Optional[BitsAndBytesConfig]:
    """Return bitsandbytes config for optional 4-bit/8-bit loading."""
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


def resolve_dtype(dtype_choice: str) -> Optional[torch.dtype]:
    """Translate CLI dtype choices into torch dtypes."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_choice)


def run_training(args: argparse.Namespace) -> None:
    """Primary entrypoint: build dataset, reward, and launch GRPO fine-tuning."""
    train_dataset: Dataset
    test_samples: List[Dict] = []
    metadata_map: Dict[str, Dict] = {}
    prompt_to_sample: Dict[str, str] = {}
    eval_dataset: Optional[Dataset] = None

    dataset_path = Path(args.grpo_dataset_json)
    prompt_path = Path(args.prompt_template)

    train_dataset, test_samples, metadata_map, prompt_to_sample = prepare_plan_datasets(dataset_path, prompt_path, args.train_ratio, args.split_seed)
    reward_fn = build_plan_reward_function(metadata_map, prompt_to_sample)
    
    logger.info(
        "Loaded custom dataset %s with %d training samples and %d test samples",
        dataset_path,
        len(train_dataset),
        len(test_samples),
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eval_batch_size = args.eval_batch_size or args.train_batch_size
    evaluation_kwargs: Dict[str, object] = {}
    if test_samples:
        evaluation_kwargs.update(
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy="steps",
            eval_steps=max(1, args.eval_steps),
        )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        report_to=[],
        bf16=args.bf16,
        fp16=args.fp16,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        **evaluation_kwargs,
    )

    quant_config = build_quantization_config(args.load_in_4bit, args.load_in_8bit)
    dtype = resolve_dtype(args.torch_dtype)

    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    chat_template_available = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    if chat_template_available:
        logger.info("Applying chat template formatting to training prompts.")

        def _format_example(example: Dict) -> Dict:
            example["prompt"] = format_prompt_with_chat_template(example["prompt"], tokenizer)
            return example

        train_dataset = train_dataset.map(_format_example)

        prompt_to_sample.clear()
        for record in train_dataset:
            sample_id = record["sample_id"]
            formatted_prompt = record["prompt"]
            prompt_to_sample[formatted_prompt] = sample_id
            metadata = metadata_map.get(sample_id)
            if metadata:
                metadata["prompt"] = formatted_prompt

    if not chat_template_available:
        logger.warning("Tokenizer missing chat template; training will use raw prompts.")

    for sample in test_samples:
        record = sample["record"]
        sample_id = record["sample_id"]
        formatted_prompt = format_prompt_with_chat_template(record["prompt"], tokenizer)
        record["prompt"] = formatted_prompt
        metadata = metadata_map.get(sample_id)
        if metadata:
            metadata["prompt"] = formatted_prompt
        prompt_to_sample.setdefault(formatted_prompt, sample_id)

    if test_samples:
        eval_records = [sample["record"] for sample in test_samples]
        eval_dataset = Dataset.from_list(eval_records)
        logger.info(
            "Configured validation with %d samples; running every %d training step(s).",
            len(eval_dataset),
            max(1, args.eval_steps),
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=build_lora_config(args),
    )

    logger.info("Starting GRPO training with %s", model_name)
    trainer.train()
    logger.info("Training complete; checkpoints saved to %s", args.output_dir)


def build_lora_config(args: argparse.Namespace):
    """Construct a LoRAConfig with user-provided overrides."""
    from peft import LoraConfig

    target_modules = (
        args.lora_target_modules.split(",")
        if args.lora_target_modules
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
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def normalize_plan_string(plan_text: str) -> str:
    """Return a canonical string representation with single-quoted tools."""
    parsed = try_parse_plan(plan_text.strip())
    if parsed is None:
        return plan_text.strip()
    return repr(parsed)




# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Build the CLI for GRPO training workflows."""
    parser = argparse.ArgumentParser(description="GRPO training harness for plan generation.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a GRPO policy with optional custom dataset.")
    train_parser.add_argument("--grpo-dataset-json", help="Path to grpo_dataset_seq.json for custom training.")
    train_parser.add_argument("--prompt-template", default="catp_experiments/prompts/prompt_seq.txt", help="Prompt template used for custom dataset.")
    train_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio when using custom dataset.")
    train_parser.add_argument("--split-seed", type=int, default=42, help="Random seed for task split.")
    train_parser.add_argument("--output-dir", default="grpo-qwen2.5-7B", help="Directory to save checkpoints.")
    train_parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device train batch size.")
    train_parser.add_argument("--eval-batch-size", type=int, help="Per-device eval batch size (defaults to train batch size).")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    train_parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    train_parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs.")
    train_parser.add_argument("--max-prompt-length", type=int, default=8000, help="Maximum prompt token length.")
    train_parser.add_argument("--max-completion-length", type=int, default=256, help="Maximum completion token length.")
    train_parser.add_argument("--num-generations", type=int, default=8, help="Number of completions sampled per prompt (GRPO K).")
    train_parser.add_argument("--eval-steps", type=int, default=1, help="Run evaluation every N training steps.")
    train_parser.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Torch dtype for base model.")
    train_parser.add_argument("--device-map", default="auto", help="Device map for model loading.")
    train_parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization.")
    train_parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization.")
    train_parser.add_argument("--bf16", action="store_true", help="Enable bf16 training in GRPO config.")
    train_parser.add_argument("--fp16", action="store_true", help="Enable fp16 training in GRPO config.")
    train_parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Checkpoint saving strategy for GRPOTrainer.",
    )
    train_parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Number of update steps between checkpoints when --save-strategy=steps.",
    )
    train_parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank.")
    train_parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha.")
    train_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    train_parser.add_argument("--lora-target-modules", help="Comma-separated LoRA target modules.")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    return args

def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_training(args)
    else:
        raise ValueError(f"Unknown command {args.command}")

if __name__ == "__main__":
    main()
