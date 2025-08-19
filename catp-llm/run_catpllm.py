import argparse
import os
import pickle
import json
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from munch import munchify
from pprint import pprint

from src.config import GlobalToolConfig, GlobalPathConfig, GlobalTaskConfig, PRETRAINED_LLM_DIR
from src.catpllm.data.plan_dataset import PlanDataset
from src.catpllm.pipeline.train import Trainer
from src.catpllm.pipeline.test import test_fn
from src.catpllm.model import OfflineRLPolicy, TokenEncoder
from src.catpllm.model.llm import peft_model
from src.catpllm.utils.llm_utils import load_llm
from src.catpllm.utils.utils import set_random_seed, save_model, load_model, load_yaml_config


def train(args, policy, plan_dataset, checkpoint_dir, best_model_dir):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in policy.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in policy.llm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, policy, optimizer, loss_fn, plan_dataset, args.context_len, args.llm_device,
                      lr_scheduler=lr_scheduler, grad_accum_steps=args.grad_accum_steps, num_plans=args.num_plans, 
                      fast_eval=args.fast_eval, scheduled_sampling_rate=args.scheduled_sampling_rate)

    best_eval_loss = float('inf')  # for fast evaluation

    total_train_losses = []
    for epoch in range(args.epochs):
        train_logs, train_losses = trainer.train(num_episodes=args.episodes_per_epoch)
        total_train_losses.extend(train_losses)
        print('=' * 20, f'Training Epoch #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        # evaluate 
        save_checkpoint_dir = os.path.join(checkpoint_dir, str(epoch))
        os.makedirs(save_checkpoint_dir, exist_ok=True)
        save_model(args, policy, save_checkpoint_dir)
        print('Checkpoint saved at:', save_checkpoint_dir)
        if args.fast_eval:
            eval_logs, _ = trainer.fast_eval_fn()
            if best_eval_loss > eval_logs['training/fast_eval_loss_mean']:
                best_eval_loss = eval_logs['training/fast_eval_loss_mean']
                save_model(args, policy, best_model_dir)
                print('Best model saved at:', best_model_dir)
            eval_logs['best_result'] = best_eval_loss
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)

    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test(args, policy, model_dir, results_dir, dataset_info, process_reward_fn):
    if os.path.exists(model_dir):
        policy = load_model(args, policy, model_dir)
        print('Load model from:', model_dir)
    else:
        print('Model dir does not exist, skip loadding model from:', model_dir)
        
    target_return = dataset_info.max_return * args.target_return_scale
    test_logs, valid_plans, invalid_plans = test_fn(args, policy, dataset_info, target_return, process_reward_fn)
    valid_plans_path = os.path.join(results_dir, 'valid_plans.json')
    json.dump(valid_plans, open(valid_plans_path, 'w', encoding='utf-8'), indent=4)
    invalid_plans_path = os.path.join(results_dir, 'invalid_plans.json')
    json.dump(invalid_plans, open(invalid_plans_path, 'w', encoding='utf-8'), indent=4)
    print('Test time:', test_logs['time/testing'])
    print('Results saved at:\n', 'valid plans - ', valid_plans_path, '\n', 'invalid plans - ', invalid_plans_path)


def main(args):
    assert args.train_plan_pool is not None, "Please specify plan pools for training."

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create plan dataset
    plan_pools = pickle.load(open(args.train_plan_pool, 'rb'))
    plan_dataset = PlanDataset(plan_pools, args.alpha, args.gamma, args.scale, args.context_len)

    dataset_info = munchify(plan_dataset.dataset_info)
    pprint(dataset_info)

    # 3. create policy model
    # 3.1 load llm
    llm, tokenizer, llm_config = load_llm(args.llm, os.path.join(PRETRAINED_LLM_DIR, args.llm))  
    llm = llm.to(args.llm_device)
    if args.rank != -1:
        llm = peft_model(llm, args.llm, args.rank)

    # 3.2 create token encoder
    llm_embed_dim = llm.get_embed_dim()
    encoder = TokenEncoder(GlobalToolConfig.max_num_tokens, llm_embed_dim, device=args.llm_device).to(args.llm_device)

    # 3.3 create policy
    num_tool_tokens = len(GlobalToolConfig.tool_token_vocabulary.keys()) + 2
    num_dependency_tokens = len(GlobalToolConfig.dependency_token_vocabulary.keys()) + 2
    policy = OfflineRLPolicy(encoder, tokenizer, llm, llm_embed_dim, num_tool_tokens, num_dependency_tokens, 
                             GlobalToolConfig.max_num_tokens, device=args.llm_device, max_window_size=args.context_len)

    # 4. handling directory and path
    config_file_name = 'config_' + args.config_file.split('/')[-1].split('.')[0]
    models_dir = os.path.join(GlobalPathConfig.finetune_path, 'catpllm', config_file_name)
    results_dir = os.path.join(GlobalPathConfig.result_path, 'plan_info', 'catpllm', config_file_name)
    checkpoint_dir = os.path.join(models_dir, 'checkpoint')
    best_model_dir = os.path.join(models_dir, 'best_model')

    # 5. start training/testing
    def process_reward(reward,
                       max_reward=dataset_info.max_reward,
                       min_reward=dataset_info.min_reward,
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return reward

    torch.backends.cudnn.benchmark = True
    if args.train:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        train(args, policy, plan_dataset, checkpoint_dir, best_model_dir)
    if args.test:
        os.makedirs(results_dir, exist_ok=True)
        model_dir = args.load_model_dir if args.load_model_dir is not None else best_model_dir
        if args.test_task_list is None:
            args.test_task_list = GlobalTaskConfig.default_test_seq_tasks
        test(args, policy, model_dir, results_dir, dataset_info, process_reward)


if __name__ == "__main__":
    # create the parser object
    parser = argparse.ArgumentParser(description="Process parameters for running benchmark tasks")
    parser.add_argument("--config_file", type=str, help="Configuration file for running CATP-LLM.")  # specify where to run llm.
    parser.add_argument("--llm", type=str, help="Name of the llm (format: llm-size, e.g., opt-350m)", default="opt-350m")
    parser.add_argument("--llm_device", type=str, help="Device for LLM", default='cuda' if torch.cuda.is_available() else 'cpu')  # specify where to run llm.

    # arguments for other settings
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--test_task_list", type=int, nargs='+')

    
    # parse the arguments
    args = parser.parse_args()
    # args.config_file = 'src/catpllm/data/config_data/default_debug.yaml'  # debug
    assert args.config_file is not None
    
    config = load_yaml_config(args.config_file)
    args = munchify(vars(args))
    args.update(config)

    pprint(args)
    main(args)
