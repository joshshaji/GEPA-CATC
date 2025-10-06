from modules import PlanGenerator
import os
import numpy as np
from PIL import Image
import dspy
import json

dataset_path = "/Users/mdamarap/GEPA-CATC/catp-llm/dataset"
task_descriptions_path = os.path.join(dataset_path, "task_descriptions.txt")
system_prompt_path = os.path.join(dataset_path, "/Users/mdamarap/GEPA-CATC/catp_gepa/system_prompt_non_seq_gepa.txt")

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

lm = dspy.LM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=anthropic_api_key,
    temperature=0,
)

dspy.configure(lm=lm)
program = PlanGenerator()

def read_task_by_index(index: int) -> str:
    with open(task_descriptions_path, "r", encoding="utf-8") as f:
        tasks = [line.rstrip("\n") for line in f]
    return tasks[index]

def get_tool_prices(image_size: tuple[int, int]):
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

def output_json(
    task_list: list[int],
    file_name: str) -> None:
    results: Dict[str, Dict] = {}
    for task in task_list:
        task_results: Dict[str, Dict] = {}
        for i in range(100):
            print("Task: ", task, "Image: ", i)
            img_path = os.path.join(dataset_path, str(task), "inputs", "images", f"{i}.jpg")
            with Image.open(img_path) as img:
                image_size = img.size  # (width, height)
            
            custom_tool_prices = get_tool_prices(image_size)
            task_query = read_task_by_index(int(task))
            plan = program(task_query, custom_tool_prices)

            task_results[str(i)] = {"plan": plan.get("plan_json")}
            print(task_query)
            print(plan.get("plan_json"))
        results[str(task)] = task_results

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    default_test_seq_tasks = [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81]
    default_test_nonseq_tasks = [200, 202, 203, 204, 205, 207, 209, 212, 215, 218, 219, 221]

    #output_json(default_test_seq_tasks, "/Users/mdamarap/GEPA-CATC/catp_gepa/results/claude_sonnet4_seq.json")
    output_json(default_test_nonseq_tasks, "/Users/mdamarap/GEPA-CATC/catp_gepa/results/claude_sonnet4_nonseq_gepa.json")
