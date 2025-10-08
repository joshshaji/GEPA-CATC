import json

def count_matching_plans(file1, file2):
    # Load both JSONs
    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    matches = 0
    total = 0

    # Compare plan strings
    for outer_key in data1:
        if outer_key in data2:
            for inner_key in data1[outer_key]:
                if inner_key in data2[outer_key]:
                    total += 1
                    plan1 = data1[outer_key][inner_key]["plan"]
                    plan2 = data2[outer_key][inner_key]["plan"]
                    if plan1 == plan2:
                        matches += 1
                    else:
                        print(f"Mismatch at {outer_key}/{inner_key}:\nPlan 1: {plan1}\nPlan 2: {plan2}")

    print(f"Total comparable outputs: {total}")
    print(f"Matching outputs: {matches}")
    print(f"Match rate: {matches / total:.2%}" if total > 0 else "No comparable outputs")

    return matches, total


if __name__ == "__main__":
    # Replace with your actual file paths
    file1 = "/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/claude_sonnet_4_seq_gepa.json"
    file2 = "/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/claude_sonnet_4_seq_gepa_2.json"
    count_matching_plans(file1, file2)
