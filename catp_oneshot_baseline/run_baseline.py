import os
from anthropic import Anthropic
from PIL import Image

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in environment")

def load_prompt(task: str, image_size: int, template_file: str = "/workspace/GEPA-CATC/catp_oneshot_baseline/prompt.txt") -> str:
    """Load prompt template and substitute task and image size."""
    with open(template_file, "r") as f:
        template = f.read()
    return template.replace("{task}", task).replace("{image_size}", str(image_size))

def read_task_by_index(index):
    file_path = "/workspace/GEPA-CATC/catp-llm/dataset/task_descriptions.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]  # remove empty lines
    return tasks[index]

def infer_llm_call(prompt: str) -> dict:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build messages API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",   # You can change to opus/haiku etc.
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    # Anthropic returns a list of content blocks; take the first text block
    return response.content[0].text if response.content else ""

if __name__ == "__main__":
    dataset = "/workspace/GEPA-CATC/catp-llm/dataset";
    # Example usage
    task = "1"
    img = "26"
    img_path = os.path.join(dataset, task, "inputs", "images", img + ".jpg")
    with Image.open(img_path) as img:
        image_size = img.size  # (width, height)
    task_str = read_task_by_index(int(task))
    print(task_str, image_size)
    prompt = load_prompt(task_str, image_size)
    result = infer_llm_call(prompt)
    print(result)