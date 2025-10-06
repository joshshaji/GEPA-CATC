import json
import sys

def convert_quotes_in_plans(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Function to recursively process the JSON structure
    def process_item(item):
        if isinstance(item, dict):
            # Process each key-value pair in the dictionary
            for key, value in item.items():
                if key == 'plan' and isinstance(value, list):
                    # Convert double quotes to single quotes in the plan array
                    for i, element in enumerate(value):
                        if isinstance(element, basestring):
                            # Replace double quotes with single quotes in tool names
                            item[key][i] = element.replace('"', "'")
                        elif isinstance(element, list):
                            # Process nested arrays (the dependency lists)
                            for j, sub_element in enumerate(element):
                                if isinstance(sub_element, basestring):
                                    item[key][i][j] = sub_element.replace('"', "'")
                else:
                    # Recursively process nested dictionaries and lists
                    process_item(value)
        elif isinstance(item, list):
            # Process each element in the list
            for element in item:
                process_item(element)
    
    # Process the entire data structure
    process_item(data)
    
    # Write the modified data back to the file
    with open(file_path, 'w') as f:
        # Use ensure_ascii=False to preserve non-ASCII characters
        json.dump(data, f, indent=2, ensure_ascii=False, encoding='utf-8')

if __name__ == "__main__":
    file_path = "/Users/mdamarap/GEPA-CATC/catp_gepa/results/claude_sonnet4_seq.json"
    convert_quotes_in_plans(file_path)
    print "Successfully updated quotes in {}".format(file_path)
