'''
Given a JSON file of plan results, this script will compute the Average QoP of the JSON file.
Basically, this script is the last step of the pipeline to evaluate the final QoP of the test set.
'''

import json
import ast
from torch.utils.data import DataLoader
from src.config import GlobalPathConfig
from src.plan import Plan
from src.data_loader import TaskDataset
from src.metrics import calculate_qop, calculate_task_score


# Enter the path for the Output JSON File to be evaluated
input_json = "/workspace/GEPA-CATC/catp-llm/sample.json"



def get_qop(task_id, img_id, plan_str):    
    # Convert plan string to Python list
    plan_list = ast.literal_eval(plan_str)
    
    # Initialize Plan
    plan = Plan(plan_list)
    task_id_int = int(task_id)
    
    # Load dataset for the task
    data_set = TaskDataset(GlobalPathConfig.data_path, task_id=task_id_int)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
    
    for batch in data_loader:
        sample_id = batch["sample_id"]
        input_data = batch["input"]
        output_data = batch["output"]
                
        if int(sample_id.item()) != int(img_id):
            continue
        
        result = plan.execute(input_data)
        if result is not None:
            cost_price = plan.price
            exec_time = plan.exec_time
            task_score = calculate_task_score(result, output_data, sequential = task_id_int < 200)
            qop = calculate_qop(task_score, cost_price)
            break  # only process first batch
    
    print("\n Task Id : {} and Image Id : {} and QoP : {} \n".format(task_id, img_id, qop))
    return qop

with open(input_json, 'r') as file:
    data = json.load(file)

qop = 0.0
count = 0
for task_id, images in data.items():
    for img_id, plan in images.items():
        plan_str = plan.get("plan")
        qop += get_qop(task_id, img_id, plan_str)
        count += 1

print("Average QoP of the Input JSON: ", qop / count)