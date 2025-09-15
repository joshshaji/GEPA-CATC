import os
import torch
import cv2
import yaml
import random
import numpy as np
from munch import munchify
from bisect import bisect_left
from src.config import GlobalToolConfig, GlobalDataConfig, GlobalMetricsConfig


def find_sink_nodes_in_plan(tool_plan, is_token=False, return_token=False):
    plan = []
    if is_token:
        for token in tool_plan:
            if token in [GlobalToolConfig.sop_token, GlobalToolConfig.eop_token, GlobalToolConfig.sod_token, GlobalToolConfig.eod_token]:
                continue
            if token < GlobalToolConfig.dependency_token_start:  # a tool token
                plan.append(GlobalToolConfig.tool_token_vocabulary_reverse[token])
            else:  # a dependency token
                if type(plan[-1]) is str:
                    plan.append([])
                # TODO: Support dependency token list
                plan[-1].append(GlobalToolConfig.dependency_token_vocabulary_reverse[token])
    else:
        plan = tool_plan

    out_degrees = {tool: 0 for tool in plan[::2]}
    for dependencies in plan[1::2]:
        for dependency_tool in dependencies:
            if dependency_tool == 'Input of Query':
                continue
            out_degrees[dependency_tool] += 1
    
    sink_nodes = []
    for tool, out in out_degrees.items():
        if out == 0:
            sink_nodes.append(tool)
    
    if return_token:
        return [GlobalToolConfig.tool_token_vocabulary[node] for node in sink_nodes]
    return sink_nodes


def get_task_and_sample_info(task_id, sample_id, data_path):
    """
    Fetch the information of the task specification and data sample.

    :param task_id: the index of the task.
    :param sample_id: the index of the sample.
    :return task_info (str), sample_info (dict).
    """
    # first fetch task information
    with open(os.path.join(data_path, 'task_description.txt')) as file:
        lines = file.readlines()
        task_info = lines[task_id].strip()
        file.close()
    
    # next fetch sample information
    if 0 <= task_id <= 104:  # task input = image
        image_path = os.path.join(data_path, str(task_id), 'inputs', 'images', f'{sample_id}.jpg')
        image = cv2.imread(image_path)
        height, width = image.shape[0], image.shape[1]
        sample_info = {'has_image': True, 'image_size': (width, height), 'has_text': False, 'text_length': None}
    elif 107 <= task_id <= 114:
        text_path = os.path.join(data_path, str(task_id), 'inputs', 'text.txt')
        with open(text_path, 'r') as file:
            lines = file.readlines()
            text = lines[sample_id].strip()
            file.close()
        sample_info = {'has_image': False, 'image_size': None, 'has_text': True, 'text_length': len(text)}
    elif 200 <= task_id <= 229:
        image_path = os.path.join(data_path, str(task_id), 'inputs', 'images', f'{sample_id}.jpg')
        image = cv2.imread(image_path)
        height, width = image.shape[0], image.shape[1]
        sample_info = {'has_image': True, 'image_size': (width, height), 'has_text': False, 'text_length': None}
    else:
        raise ValueError('Currently we only support tasks 0~104 or 107~114 or 200~229.')
    # sample_info['batch_size'] = 1
    return task_info, sample_info


def determine_sample_size(sample_info, task_id, sample_id, data_path):
    """
    Determine the size of the sample.
    Note that we categorize the sizes of data samples into k levels (index from 0 to k - 1), from the smallest to the largest.
    So this function is used to determine which level the sample belongs to.
    """
    if sample_info is None:
        sample_info = get_task_and_sample_info(task_id, sample_id, data_path)[1]
    
    if sample_info['has_image']:
        input_size = sample_info['image_size']
        input_size = input_size[0] * input_size[1]
        input_sizes = GlobalDataConfig.image_sizes
    else:
        input_size = sample_info['text_length']
        input_sizes = GlobalDataConfig.text_lengths

    index = bisect_left(input_sizes, input_size)
    if 1 <= index <= len(input_sizes) - 1:
        if abs(input_sizes[index - 1] - input_size) >= abs(input_sizes[index] - input_size):
            sample_size = index
        else:
            sample_size = index - 1
    elif index >= len(input_sizes):
        sample_size = len(input_sizes) - 1
    else:
        sample_size = index
    
    return sample_size


def process_data(states, returns, actions, timesteps, device='cpu'):
    """
    Transform the format of the data samples in the plan dataset to make them easier for further processing.
    """
    states = [states]  # List[List[List]], shape = (batch_size=1, sequence_len, x)
    returns = torch.tensor(returns, device=device, dtype=torch.float).reshape(1, -1, 1)  # shape = (1, sequence_len, 1)
    labels = torch.tensor(actions, device=device, dtype=torch.int64).reshape(1, -1)  # shape = (1, sequence_len)
    timesteps = torch.tensor(timesteps, device=device, dtype=torch.int64).reshape(1, -1)  # shape = (1, sequence_len)
    # prepare actions
    tool_token_indices = labels < GlobalToolConfig.dependency_token_start
    dependency_token_indices = labels >= GlobalToolConfig.dependency_token_start
    actions = torch.tensor(actions, device=device, dtype=torch.float).reshape(1, -1)
    actions[tool_token_indices] -= GlobalToolConfig.tool_token_start
    actions[dependency_token_indices] -= GlobalToolConfig.dependency_token_start
    actions = actions.reshape(1, -1, 1)  # shape = (1, sequence_len, 1)
    actions /= GlobalToolConfig.max_num_tokens
    return states, returns, actions, timesteps, labels


def calculate_cost_aware_reward(score, cost, alpha, min_score, max_score, 
                                min_cost, max_cost, scale=1, is_real_score=True):
    """
    Calculate rewards for cost-aware rl.
    :param scale: scale up reward
    """
    max_cost = GlobalMetricsConfig.MAX_COST
    min_score = GlobalMetricsConfig.MIN_SCORE
    if cost != GlobalMetricsConfig.cost_penalty:
        norm_score = (score - min_score) / (max_score - min_score)
        norm_cost = (cost - min_cost) / (max_cost - min_cost)
        if is_real_score:
            reward = alpha * norm_score - (1 - alpha) * norm_cost
        else:
            reward = -(1 - alpha) * norm_cost
    else:
        reward = -0.5
    return reward * scale


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def save_model(args, model, save_dir):
    if args.rank > 0:
        # save lora weights
        model.llm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_llm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # lora is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load lora weights
        model.llm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_llm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin'), map_location=args.llm_device))
    else:
        # lora is disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def load_yaml_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    config = munchify(config)
    return config


def token_plan_to_opencatp_plan(token_plan):
    """ Transfrom tokens in the tool plan into the OpenCATP form in natural language."""
    plan = []
    for token in token_plan:
        if token in [GlobalToolConfig.sop_token, GlobalToolConfig.eop_token, 
                     GlobalToolConfig.sod_token, GlobalToolConfig.eod_token]:
            continue
        if token < GlobalToolConfig.dependency_token_start:  # a tool token
            plan.append(GlobalToolConfig.tool_token_vocabulary_reverse[token])
        else:  # a dependency token
            if type(plan[-1]) is str:
                plan.append([])
            plan[-1].append(GlobalToolConfig.dependency_token_vocabulary_reverse[token])
    return plan