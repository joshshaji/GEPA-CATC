import json

# Read the input file
with open('/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/qwen2.5_7B_seq_1.json', 'r') as f:
    data = json.load(f)

# Process the data to replace double quotes with single quotes in plan strings
for task_id, task_data in data.items():
    for image_id, image_data in task_data.items():
        if 'plan' in image_data:
            # Replace double quotes with single quotes
            image_data['plan'] = image_data['plan'].replace('"', "'")

# Write the modified data back to the file
with open('/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/qwen2.5_7B_seq_1.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully converted double quotes to single quotes in the plan strings.")
