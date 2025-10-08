import json

with open("/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/claude_sonnet_4_seq_2.json") as f:
    data = json.load(f)

for outer_key, outer_val in data.items():
    for inner_key, inner_val in outer_val.items():
        inner_val["plan"] = inner_val["plan"].replace('"', "'")

with open("/Users/mdamarap/GEPA-CATC/catp_experiments/output_jsons/claude_sonnet_4_seq_2_fixed.json", "w") as f:
    json.dump(data, f, indent=2)