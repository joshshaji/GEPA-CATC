import torch
import time
from copy import copy

from src.config import GlobalToolConfig, GlobalPathConfig
from src.catpllm.model.offline_rl import TOOL_PREDICTION_MODE, DEPENDENCY_PREDICTION_MODE
from src.catpllm.utils.cost_utils import estimate_tool_price
from src.catpllm.utils.utils import get_task_and_sample_info, determine_sample_size, calculate_cost_aware_reward


def inference(args, policy, dataset_info, task_id, target_return, process_reward_fn=None, sample_ids=None, reused_plan=True):
    """
    Use the LLM policy to generate tool plans.

    :param reused_plan: Enable reuse of plans across samples under the same levels of data sizes. Though this may hurt performance, 
                        it can significantly speed up inference process.
    """
    if process_reward_fn is None:
        process_reward_fn = lambda x: x
    
    if sample_ids is None:
        sample_ids = list(range(100))

    logs = {}
    inference_start = time.time()
    tool_plans = {}
    plans_buffer = {}

    policy.eval()
    for sample_id in sample_ids:
        task_info, sample_info = get_task_and_sample_info(task_id, sample_id, data_path=GlobalPathConfig.data_path)
        sample_size = determine_sample_size(sample_info, task_id, sample_id, data_path=GlobalPathConfig.data_path)

        if reused_plan:  # the samples with the same input size level and belonging to the same task can share the same plan. this trick can speed up plan generation, though may degrade performance.
            if sample_size in plans_buffer:
                tool_plans[sample_id] = plans_buffer[sample_size]
                continue

        tool_plan = [GlobalToolConfig.sop_token]
        target_return_copy = copy(target_return)
        gamma = 1
        timestep = 0
        mode = TOOL_PREDICTION_MODE
        prev_cost = 0.
        num_generated_tokens = 0
        with torch.no_grad():
            while True:
                timestep = min(len(tool_plan) - 1, GlobalToolConfig.max_ep_len)
                token = policy.inference(tool_plan, target_return, timestep, task_id, task_info, sample_info, sample_size, mode)
                tool_plan.append(token)

                # calculate reward and update target return
                score = (1 / GlobalToolConfig.max_num_generated_tokens) if token not in [GlobalToolConfig.eop_token, GlobalToolConfig.eod_token] else 0.
                if token < GlobalToolConfig.dependency_token_start and token != GlobalToolConfig.eop_token:
                    cost = estimate_tool_price(GlobalToolConfig.tool_token_vocabulary_reverse[token], sample_size)
                    prev_cost = cost
                else:
                    cost = prev_cost
                    
                reward = calculate_cost_aware_reward(score, cost, args.alpha, dataset_info.min_score, dataset_info.max_score,
                                                     dataset_info.min_cost, dataset_info.max_cost, args.scale, is_real_score=False)
                reward = process_reward_fn(reward)
                target_return_copy -= gamma * reward
                gamma *= args.gamma

                if token == GlobalToolConfig.eop_token:
                    break
                elif token < GlobalToolConfig.dependency_token_start:
                    mode = DEPENDENCY_PREDICTION_MODE
                    tool_plan.append(GlobalToolConfig.sod_token)
                    target_return_copy -= gamma * reward
                    gamma *= args.gamma
                elif token == GlobalToolConfig.eod_token:
                    mode = TOOL_PREDICTION_MODE

                num_generated_tokens += 1
                if num_generated_tokens > GlobalToolConfig.max_num_generated_tokens:
                    break

        tool_plans[sample_id] = tool_plan
        plans_buffer[sample_size] = tool_plan
        policy.clear_cache()
    
    logs['time/inferencing'] = time.time() - inference_start
    logs['task id'] = task_id
    logs['num of sample'] = len(sample_ids)

    return tool_plans, logs
