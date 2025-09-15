import json
import argparse

from torch.utils.data import DataLoader

from src.config import GlobalPathConfig
from src.plan import Plan
from src.data_loader import TaskDataset
from src.metrics import calculate_qop, calculate_task_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default="./test_samples.json", help="Path to samples JSON")
    parser.add_argument("--task_id", type=int, default=None, help="Filter to a single task id")
    parser.add_argument("--sample_id", type=int, default=None, help="Filter to a single sample id")
    args = parser.parse_args()

    with open(args.samples, "r") as f:
        data = json.load(f)

    task_ids = list(map(int, data.keys()))
    if args.task_id is not None:
        task_ids = [args.task_id] if str(args.task_id) in data else []

    for task_id in task_ids:
        plan_info = data[str(task_id)]["plan"]
        plan = Plan(plan_info)
        data_set = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        for batch in data_loader:
            sample_id = int(batch["sample_id"])  # tensor -> int
            if args.sample_id is not None and sample_id != args.sample_id:
                continue
            input_data = batch["input"]
            output_data = batch["output"]
            result = plan.execute(input_data)
            if result is None:
                print(task_id, sample_id, "invalid plan")
            else:
                task_score = calculate_task_score(result, output_data, sequential=task_id < 200)
                cost_price = plan.price
                exec_time = plan.exec_time
                qop = calculate_qop(task_score, cost_price)
                print(task_id, sample_id, task_score, cost_price, exec_time, qop)
            # For a simple check: only evaluate one sample per task unless you explicitly loop externally
            break
    print("done")


if __name__ == "__main__":
    main()
