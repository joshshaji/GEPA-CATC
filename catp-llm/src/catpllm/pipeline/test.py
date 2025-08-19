import time
import torch

from torch.utils.data import DataLoader

from src.config import GlobalTaskConfig, GlobalPathConfig, GlobalMetricsConfig
from src.plan import Plan
from src.data_loader import TaskDataset
from src.catpllm.pipeline.inference import inference
from src.catpllm.utils.utils import token_plan_to_opencatp_plan
from src.metrics.evaluator import calculate_task_score, calculate_qop


def test_fn(args, policy, dataset_info, target_return, process_reward_fn):
    """
    Use the LLM policy to generate tool plans and evaluate the quality of the generated plans.
    """
    test_logs = {}
    test_start = time.time()

    policy.eval()
    valid_plans = {}
    invalid_plans = {}
    with torch.no_grad():
        # 1. Use the LLM policy to generate tool plans.
        task_tool_plans = {}
        test_task_samples = GlobalTaskConfig.default_test_task_samples
        for task_id in args.test_task_list:
            sample_ids = test_task_samples[str(task_id)]
            tool_plans, _ = inference(args, policy, dataset_info, task_id, target_return, process_reward_fn, sample_ids)
            task_tool_plans[task_id] = tool_plans

        policy = policy.to('cpu')  # save gpu memory

        # 2. Evaluate the generated tool plans.
        for task_id, tool_plans in task_tool_plans.items():
            sample_ids = set(tool_plans.keys())
            task_dataset = TaskDataset(GlobalPathConfig.data_path, task_id=task_id)
            dataloader = DataLoader(task_dataset, batch_size=1, shuffle=False)
            invalid_tool_plans_buffer = set()
            plan_buffer = {}
            
            valid_plans[task_id] = {}
            invalid_plans[task_id] = {}
            for idx, batch in enumerate(dataloader):
                sample_id = batch['sample_id'].item()
                if sample_id not in sample_ids:
                    continue

                tool_plan = tool_plans[sample_id]
                opencatp_plan = token_plan_to_opencatp_plan(tool_plan)
                if str(opencatp_plan) in invalid_tool_plans_buffer:
                    invalid_plans[(task_id, sample_id)] = str(opencatp_plan)
                else:
                    if str(opencatp_plan) not in plan_buffer:
                        plan = Plan(opencatp_plan)
                        plan_buffer[str(opencatp_plan)] = plan
                    else:
                        plan = plan_buffer[str(opencatp_plan)]
                    input_data = batch["input"]
                    output_data = batch["output"]
                    try:
                        result = plan.execute(input_data)
                    except torch.OutOfMemoryError:  # skip samples that are too large to be executed on the current GPU device.
                        continue
                    if result is None:  # invalid plans
                        task_score = GlobalMetricsConfig.score_penalty
                        cost_price = GlobalMetricsConfig.cost_penalty
                        exec_time = None
                        qop = None
                        invalid_tool_plans_buffer.add(str(opencatp_plan))
                        invalid_plans[task_id][sample_id] = {'plan': str(opencatp_plan), 'task_score': task_score, 'cost_price': cost_price,
                                                             'exec_time': exec_time, 'qop': qop}
                    else:
                        task_score = calculate_task_score(result, output_data, sequential=task_id < 200)
                        cost_price = plan.price
                        exec_time = plan.exec_time
                        qop = calculate_qop(task_score, cost_price)
                        valid_plans[task_id][sample_id] = {'plan': str(opencatp_plan), 'task_score': task_score, 'cost_price': cost_price,
                                                           'exec_time': exec_time, 'qop': qop}
                # break  # debug
            
    test_logs.update({
        'time/testing': time.time() - test_start,
    })
    return test_logs, valid_plans, invalid_plans

