import json

with open("/Users/mdamarap/GEPA-CATC/rtx_4090_baselines/plan_results_phase2.json", "r") as f:
    data = json.load(f)

cost_price = []

for outer_key, outer_val in data.items():
    for inner_key, inner_val in outer_val.items():
        for entry in inner_val:
            if "cost_price" in entry:
                cost_price.append(entry["cost_price"])

if cost_price:
    print("Maximum cost_price:", max(cost_price) / len(cost_price))
else:
    print("No exec_time values found")


#Maximum cost_price: 0.27939974767577036
#Maximum cost_price: 0.18291475511707217
#Maximum cost_price: 0.3209703642675919