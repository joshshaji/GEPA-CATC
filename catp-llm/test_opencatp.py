import json

from torch.utils.data import DataLoader

from src.config import GlobalPathConfig
from src.plan import Plan
from src.data_loader import TaskDataset
from src.metrics import calculate_qop, calculate_task_score


with open("./test_samples.json", "r") as f:
    data = json.load(f)
    
    
for task_id in data.keys():
    plan = data[task_id]["plan"]
    plan = Plan(plan)
    task_id = int(task_id)
    data_set = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
    for batch in data_loader:
        sample_id = batch["sample_id"]
        input_data = batch["input"]
        output_data = batch["output"]
        result = plan.execute(input_data)
        if result is None:
            pass
        else:
            task_score = calculate_task_score(result, output_data, sequential=task_id < 200)
            cost_price = plan.price
            exec_time = plan.exec_time
            qop = calculate_qop(task_score, cost_price)
            print(task_id, task_score, cost_price, exec_time, qop)
        break
print("done")