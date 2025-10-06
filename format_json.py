import json
import sys

def format_json_file(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each entry to format the plan as a string
    for task_id, task_data in data.items():
        for img_id, img_data in task_data.items():
            if 'plan' in img_data and isinstance(img_data['plan'], list):
                # Convert the plan list to a string representation
                plan_list = img_data['plan']
                # Create a new list to handle nested lists properly
                formatted_plan = []
                for item in plan_list:
                    if isinstance(item, list):
                        formatted_plan.append(str(item))
                    else:
                        formatted_plan.append("'" + str(item) + "'")
                # Join with commas and wrap in brackets, then remove 'u' prefixes
                plan_str = '[' + ', '.join(formatted_plan) + ']'
                plan_str = plan_str.replace("u'", "'")
                img_data['plan'] = plan_str
    
    # Write the formatted data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python format_json.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    format_json_file(input_file, output_file)
    print("Formatted JSON has been written to " + output_file)
