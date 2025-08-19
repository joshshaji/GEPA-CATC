import numpy as np
import torch
import time

from munch import Munch
from tqdm import tqdm

from src.config import GlobalToolConfig, GlobalPathConfig
from src.catpllm.model.offline_rl import TOOL_PREDICTION_MODE, DEPENDENCY_PREDICTION_MODE
from src.catpllm.utils.utils import process_data, get_task_and_sample_info, determine_sample_size


MINITERS, MAXINTERVAL = 20, 200  # for tqdm


class Trainer:
    def __init__(self, args, policy, optimizer, loss_fn, plan_dataset, context_window, device, 
                 batch_size=1, grad_accum_steps=1, lr_scheduler=None, num_plans=-1, fast_eval=False,
                 scheduled_sampling_rate=None):
        """
        :param num_plans: number of plans sampled from the dataset for training.
        :param fast_eval: enable fast evaluation or not. if enabled, we will split some proportions of plans (10% by default, no more than 500)
                          from the training dataset for evaluation.
        :param scheduled_sampling_rate: rate for scheduled sampling.
        """
        self.args = args
        self.policy = policy
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.plan_dataset = plan_dataset
        self.context_window = context_window
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        self.fast_eval = fast_eval
        self.scheduled_sampling_rate = scheduled_sampling_rate
        
        self.dataset_size = len(plan_dataset)
        self.dataset_indices = list(range(self.dataset_size))

        self.num_plans = num_plans
        if self.num_plans == -1:
            self.num_plans = self.dataset_size
        self.train_num_plans = self.num_plans
        self.train_dataset_indices = np.random.choice(self.dataset_indices, self.train_num_plans, replace=False)
        self.index = 0

        if fast_eval:
            self.eval_num_plans = min(int(self.num_plans * 0.25), 500)
            self.eval_dataset_indices = self.train_dataset_indices[-self.eval_num_plans:]
            self.train_num_plans = self.num_plans - self.eval_num_plans
            self.train_dataset_indices = self.train_dataset_indices[:-self.eval_num_plans]

        self.dataset_info = Munch(plan_dataset.dataset_info)

    def train(self, num_episodes, shuffle=True):
        total_train_losses = []
        logs = dict()

        train_start = time.time()
        self.policy.train()
        tbar = tqdm(range(num_episodes), miniters=MINITERS, maxinterval=MAXINTERVAL, desc='Training:')
        for episode in tbar:
            if self.index == 0 and shuffle:
                np.random.shuffle(self.train_dataset_indices)
            idx = self.train_dataset_indices[self.index]
            self.index = (self.index + 1) % self.train_num_plans
            
            states, returns, actions, timesteps, info = self.plan_dataset[idx]
            states, returns, actions, timesteps, labels = process_data(states, returns, actions, timesteps, self.device)
            task_id, sample_id = info['task_id'], info['sample_id']
            task_info, sample_info = get_task_and_sample_info(task_id, sample_id, data_path=GlobalPathConfig.data_path)
            sample_size = determine_sample_size(sample_info, task_id, sample_id, data_path=GlobalPathConfig.data_path)

            if self.scheduled_sampling_rate > 0:
                if np.random.rand() > self.scheduled_sampling_rate:
                    train_losses = self._teacher_forcing(states, returns, actions, timesteps, labels, task_info,
                                                         sample_info, sample_size)
                else:
                    train_losses = self._auto_regressive(states, returns, actions, timesteps, labels, task_info,
                                                         sample_info, sample_size, update_model=True)
            else:
                train_losses = self._teacher_forcing(states, returns, actions, timesteps, labels, task_info,
                                                         sample_info, sample_size)         
            total_train_losses.extend(train_losses)
            if episode % MINITERS == 0:
                tbar.set_description(f'Training: mean train loss {np.mean(total_train_losses):>9f}')       

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(total_train_losses)
        logs['training/train_loss_std'] = np.std(total_train_losses)

        return logs, total_train_losses

    def fast_eval_fn(self):
        assert self.fast_eval, "fast_eval not enabled. please set fast_eval=True."
        total_eval_losses = []
        logs = dict()

        eval_start = time.time()
        self.policy.eval()
        with torch.no_grad():
            tbar = tqdm(range(self.eval_num_plans), miniters=MINITERS, maxinterval=MAXINTERVAL, desc='Fast Evaluation:')
            for i in tbar:
                idx = self.eval_dataset_indices[i]
                states, returns, actions, timesteps, info = self.plan_dataset[idx]
                states, returns, actions, timesteps, labels = process_data(states, returns, actions, timesteps, self.device)
                task_id, sample_id = info['task_id'], info['sample_id']
                task_info, sample_info = get_task_and_sample_info(task_id, sample_id, data_path=GlobalPathConfig.data_path)
                sample_size = determine_sample_size(sample_info, task_id, sample_id, data_path=GlobalPathConfig.data_path)

                eval_losses = self._auto_regressive(states, returns, actions, timesteps, labels, task_info, sample_info,
                                                    sample_size, update_model=False)
                total_eval_losses.extend(eval_losses)
                if i % MINITERS == 0:
                    tbar.set_description(f'Fast Evaluation: mean eval loss {np.mean(total_eval_losses):>9f}')       

        logs['time/fast_eval'] = time.time() - eval_start
        logs['training/fast_eval_loss_mean'] = np.mean(total_eval_losses)
        logs['training/fast_eval_loss_std'] = np.std(total_eval_losses)

        return logs, total_eval_losses

    def _teacher_forcing(self, states, returns, actions, timesteps, labels, task_info, sample_info, sample_size):
        """ 
        Pipeline for teacher forcing. 
        """
        episode_length = len(states[0])
        losses = []
        num_iters = max(episode_length - self.context_window, 1)
        for i in range(num_iters):
            # prepare data
            batch_states = [states[0][i:i + self.context_window]]
            batch_returns = returns[:, i:i + self.context_window]
            batch_actions = actions[:, i:i + self.context_window]
            batch_timesteps = timesteps[:, i:i + self.context_window]
            batch_labels = labels[:, i:i + self.context_window]

            # feed data to the policy model
            logits, gt = self.policy(batch_states, batch_returns, batch_actions, batch_timesteps, batch_labels, 
                                        task_info, sample_info, sample_size)
            logits = logits.permute(0, 2, 1)
            loss = self.loss_fn(logits, gt)

            # perform gradient accumulation update
            loss = loss / self.grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
            if ((i + 1) % self.grad_accum_steps == 0) or (i + 1 == num_iters):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            losses.append(loss.item())
        return losses

    def _auto_regressive(self, states, returns, actions, timesteps, labels, task_info, sample_info, sample_size, update_model=True):
        """ 
        Pipeline for auto-regressive. 
        """
        episode_length = len(states[0])
        logits, gt = [], []
        for i in range(episode_length):
            # prepare data
            cur_state = states[0][i]
            cur_return = returns[0, i]
            cur_action = actions[0, i]
            cur_timestep = timesteps[0, i]
            cur_label = labels[0, i]

            if cur_label < GlobalToolConfig.dependency_token_start:
                mode = TOOL_PREDICTION_MODE
                gt.append(cur_label - GlobalToolConfig.tool_token_start)
            else:
                mode = DEPENDENCY_PREDICTION_MODE
                gt.append(cur_label - GlobalToolConfig.dependency_token_start)
            
            token, action_logits = self.policy(cur_state, cur_return, cur_action, cur_timestep, cur_label, 
                                                task_info, sample_info, sample_size, teacher_forcing=False, mode=mode)
            logits.append(action_logits)
        logits = torch.stack(logits).unsqueeze(0)
        gt = torch.stack(gt).unsqueeze(0)
        logits = logits.permute(0, 2, 1)
        loss = self.loss_fn(logits, gt)
        self.policy.clear_cache()

        if update_model:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return [loss.item()]
