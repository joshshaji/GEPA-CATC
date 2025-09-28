import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


# ----------------------------
# Utility Functions - Change these when running on Pod
# ----------------------------

dataset_path = "/Users/mdamarap/GEPA-CATC/catp-llm/dataset"
task_descriptions_path = os.path.join(dataset_path, "task_descriptions.txt")
prompt_path = "/Users/mdamarap/GEPA-CATC/catp_zeroshot_baseline/prompt.txt"

# Choose provider dynamically
provider_name = "anthropic"

# ----------------------------
# LLM Provider Abstraction
# ----------------------------

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def infer(self, prompt: str) -> str:
        """Send prompt to the LLM and return text response."""
        pass


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
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3-70b-chat-hf"):
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


def get_tool_prices(image_size: Tuple[int, int]):
    # Reference image sizes (areas)
    image_sizes = [490*402, 582*578, 954*806, 1921*2624]

    # Tool prices for each reference size
    tool_prices = {
        'image_colorization': [0.02064448408770396, 0.030365631874341062, 0.03283418516591483, 0.20662706942016817],
        'image_captioning': [0.07681041316118693, 0.0789306347162341, 0.0856651295420851, 0.15344758910825465],
        'image_classification': [0.0018386416371498659, 0.002039574896674113, 0.001632744822041972, 0.003468827901294199],
        'image_deblurring': [0.044596890616843535, 0.0763150490990877, 0.1770620811739597, 1.1910876856898132],
        'image_denoising': [0.03310938232453873, 0.05553857351136441, 0.126780926718445, 0.8438447906436202],
        'image_super_resolution': [0.12710071314082524, 0.17339502868541215, 0.3204404321613257, 1.8004475979612815],
        'machine_translation': [0.0514501296757946, 0.07495833137293198, 0.17899568650893222, 0.1265572219738673],
        'object_detection': [0.008221186114819342, 0.007555937255849043, 0.007457512004008459, 0.010285611128207868],
    }

    # Input area
    input_area = image_size[0] * image_size[1]

    # Cosine similarity in 1D reduces to min/max
    sims = np.array([min(input_area, ref) / max(input_area, ref) for ref in image_sizes])

    # Normalize
    sims /= sims.sum()
    # Weighted average of tool prices
    custom_tool_prices = {
        tool: float(np.dot(prices, sims))
        for tool, prices in tool_prices.items()
    }

    return custom_tool_prices
    
def load_prompt(task: str, custom_tool_prices: Dict[str, float], template_file: str = prompt_path) -> str:
    """Load prompt template and substitute task and image size."""
    with open(template_file, "r") as f:
        template = f.read()
    return template.replace("{task}", task).replace("{tool_prices}", json.dumps(custom_tool_prices, indent=2))


def read_task_by_index(index: int) -> str:
    with open(task_descriptions_path, "r", encoding="utf-8") as f:
        tasks = [line.rstrip("\n") for line in f]
    return tasks[index]


def output_json(
    task_list: List[int],
    file_name: str,
    provider: BaseLLMProvider,
) -> None:
    results: Dict[str, Dict] = {}
    for task in task_list:
        task_results: Dict[str, Dict] = {}
        for i in range(2):
            img_path = os.path.join(dataset_path, str(task), "inputs", "images", f"{i}.jpg")
            with Image.open(img_path) as img:
                image_size = img.size  # (width, height)

            task_str = read_task_by_index(int(task))

            custom_tool_prices = get_tool_prices(image_size)

            prompt = load_prompt(task_str, custom_tool_prices)
            result = provider.infer(prompt)

            task_results[str(i)] = {"plan": result}

            print(result)
        results[str(task)] = task_results

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    default_test_seq_tasks = [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81]
    default_test_nonseq_tasks = [200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221]

    if provider_name == "anthropic":
        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider_name == "openai":
        provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    elif provider_name == "together":
        provider = TogetherProvider(api_key=os.environ["TOGETHER_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    output_json(default_test_seq_tasks, "/Users/mdamarap/GEPA-CATC/catp_zeroshot_baseline/baseline_seq.json", provider)
    output_json(default_test_nonseq_tasks, "/Users/mdamarap/GEPA-CATC/catp_zeroshot_baseline/baseline_nonseq.json", provider)