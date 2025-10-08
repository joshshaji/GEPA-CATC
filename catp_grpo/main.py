# pip install -U trl transformers accelerate datasets peft
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# 1) tiny prompt-only dataset (any prompt column works)
dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:2000]")

# 2) toy reward: +1 if model outputs <think>...</think><answer>...</answer>
import re
def format_reward(completions, **kwargs):
    patt = r"^<think>.*?</think><answer>.*?</answer>$"
    texts = [c[0]["content"] for c in completions]
    return [1.0 if re.match(patt, t, flags=re.S) else 0.0 for t in texts]

# 3) training config (keep small to verify loop; scale later)
cfg = GRPOConfig(
    output_dir="grpo-demo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_prompt_length=512,
    max_completion_length=256,
    num_generations=4,
    report_to=[],      
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",  # pick a small instruct model first
    args=cfg,
    train_dataset=dataset,
    reward_funcs=format_reward,
    # peft_config=...                # add a LoRA config when you scale up
)

trainer.train()
