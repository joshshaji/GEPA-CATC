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
input_json = "/workspace/GEPA-CATC/catp_results/llama37B_nonseq_gepa.json"

# Output JSON file
output_file = "/workspace/GEPA-CATC/catp_results/results_llama37B_nonseq_gepa.json"


def get_task_results(task_id, img_id, plan_str):    
    try:
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
        
        print("\n Task Id : {} and Image Id : {} and QoP : {} and Task Score : {} and Cost Price : {} \n".format(task_id, img_id, qop, task_score, cost_price))
        return {
            "task_id": task_id,
            "image_id": img_id,
            "qop": qop,
            "task_score": task_score,
            "cost_price": cost_price
        }
    except Exception as e:
        print(f"Error processing Task ID: {task_id}, Image ID: {img_id} - {str(e)}")
        return None

with open(input_json, 'r') as file:
    data = json.load(file)

# Create output data structure matching input format
output_data = {}
failed_tasks = []
qop_sum = 0.0
count = 0
failed_count = 0

for task_id, images in data.items():
    output_data[task_id] = {}
    
    for img_id, plan in images.items():
        plan_str = plan.get("plan")
        task_result = get_task_results(task_id, img_id, plan_str)
        
        if task_result is not None:
            # Add the original plan data with additional fields
            output_data[task_id][img_id] = {
                "plan": plan.get("plan"),
                "qop": task_result["qop"],
                "task_score": task_result["task_score"],
                "cost_price": task_result["cost_price"]
            }
            qop_sum += task_result["qop"]
            count += 1
        else:
            failed_tasks.append({"task_id": task_id, "image_id": img_id})
            failed_count += 1

# Calculate average QoP
average_qop = (qop_sum) / (count - failed_count)


with open(output_file, 'w') as file:
    json.dump(output_data, file, indent=2)

print(f"Results saved to {output_file}")
print(f"Average QoP of the Input JSON: {average_qop}")
print(f"Total successful tasks: {count}")
print(f"Total failed tasks: {failed_count}")

if failed_count > 0:
    print("\nFailed tasks:")
    for failed_task in failed_tasks:
        print(f"  Task ID: {failed_task['task_id']}, Image ID: {failed_task['image_id']}")