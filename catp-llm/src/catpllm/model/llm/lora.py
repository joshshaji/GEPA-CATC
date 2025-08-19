import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"],
    'qwen2.5': ["q_proj", "v_proj"],
}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_target_modules(llm_type):
    if 'llama' in llm_type:
        return TARGET_MODULES['llama']
    elif 'llava' in llm_type:
        return TARGET_MODULES['llava']
    elif 'mistral' in llm_type:
        return TARGET_MODULES['mistral']
    elif 'opt' in llm_type:
        return TARGET_MODULES['opt']
    elif 'gpt2' in llm_type:
        return TARGET_MODULES['gpt2']
    elif 't5-lm' in llm_type:
        return TARGET_MODULES['t5-lm']
    elif 'qwen2.5' in llm_type:
        return TARGET_MODULES['qwen2.5']
    return None


def peft_model(llm, llm_type, rank, print_trainable=False, task_type=TaskType.FEATURE_EXTRACTION):
    for param in llm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=get_target_modules(llm_type),
        lora_dropout=0.05,
        bias="none",
        task_type=task_type
    )

    model = get_peft_model(llm, config)
    if print_trainable:
        print_trainable_parameters(model)
    return model
