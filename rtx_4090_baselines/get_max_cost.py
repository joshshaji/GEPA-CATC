import json

def get_max_cost(file):
    with open(file, "r") as f:
        data = json.load(f)

    cost_price = []

    for outer_key, outer_val in data.items():
        for inner_key, inner_val in outer_val.items():
            for entry in inner_val:
                if "cost_price" in entry:
                    cost_price.append(entry["cost_price"])
    return max(cost_price)


phase1, phase2, phase3 = get_max_cost("/Users/mdamarap/GEPA-CATC/rtx_4090_baselines/plan_results_phase1.json"), get_max_cost("/Users/mdamarap/GEPA-CATC/rtx_4090_baselines/plan_results_phase2.json"), get_max_cost("/Users/mdamarap/GEPA-CATC/rtx_4090_baselines/plan_results_phase3.json")

#Maximum cost_price: 0.27939974767577036
#Maximum cost_price: 0.18291475511707217
#Maximum cost_price: 0.3209703642675919

print(max(phase1, phase2, phase3)) # 0.3209703642675919