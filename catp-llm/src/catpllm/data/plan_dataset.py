from torch.utils.data import Dataset
from src.config import GlobalToolConfig, GlobalMetricsConfig
from src.catpllm.utils.cost_utils import calc_plan_price
from src.catpllm.utils.utils import calculate_cost_aware_reward


def discount_returns(rewards, gamma, scale):
    returns = [0. for _ in range(len(rewards))]
    returns[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        returns[i] = rewards[i] + gamma * returns[i + 1]
    for i in range(len(returns)):
        returns[i] /= scale  # scale down return
    return returns


class PlanDataset(Dataset):
    """
    A dataset class that wraps the plan sequence pool.
    """
    def __init__(self, plan_pools, alpha=0.5, gamma=1., scale=1, max_length=1):
        """
        :param plan_pools: the plan pools
        :param alpha: the weight parameter to balance the performance scores and execution costs
        :param gamma: the reward discounted factor
        :param scale: the factor to scale the reward
        :param max_length: the K value in Decision Transformer, see the paper for details.
        """
        self.dataset_info = {}
        self.max_score, self.min_score = float('-inf'), float('inf')
        self.max_cost, self.min_cost = float('-inf'), float('inf')
        self.max_plan_length = 0

        if type(plan_pools) is not list:
            plan_pools = [plan_pools]
        self.plan_pools = plan_pools
        self.plan_dataset = self._preprocess_plan_pool()
        self.plan_num = len(self.plan_dataset)

        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
        self.max_length = max_length
        self.max_reward, self.min_reward = float('-inf'), float('inf')
        self.all_states, self.all_actions, self.all_rewards = self._process_plan_dataset()
        # self._normalize_rewards()
        self.max_return, self.min_return = float('-inf'), float('inf')
        self.max_timestep, self.min_timestep = 0, 100000
        self.all_returns, self.all_timesteps = self._compute_returns()

    def _preprocess_plan_pool(self):
        """
        Preprocess the plan pool for more efficient sub-sequent processing, including:
        1. Scale each plan in the pool as a token list.
        2. Scale each cost atribute as a list.
        3. Scale each score as a list.
        """
        plan_dataset = []
        for plan_pool in self.plan_pools:
            for task_id, samples in plan_pool.plans.items():
                for sample_id in samples.keys():
                    if isinstance(plan_pool.scores[task_id][sample_id], tuple):
                        plan_pool.plans[task_id][sample_id] = [plan_pool.plans[task_id][sample_id]]
                        plan_pool.tools_exec_time[task_id][sample_id] = [plan_pool.tools_exec_time[task_id][sample_id]]
                        plan_pool.tools_cpu_mem[task_id][sample_id] = [plan_pool.tools_cpu_mem[task_id][sample_id]]
                        plan_pool.tools_gpu_mem[task_id][sample_id] = [plan_pool.tools_gpu_mem[task_id][sample_id]]
                        plan_pool.scores[task_id][sample_id] = [plan_pool.scores[task_id][sample_id]]
                    plans=plan_pool.plans[task_id][sample_id]
                    for n in range(len(plans)):
                        plan = plan_pool.plans[task_id][sample_id][n]
                        score = plan_pool.scores[task_id][sample_id][n]
                        if score[1]=='vit_score':
                            score=(score[0]/100,score[1])
                        tools_exec_time = [time for time in plan_pool.tools_exec_time[task_id][sample_id][n]]
                        tools_cpu_short_term_mem = plan_pool.tools_cpu_mem[task_id][sample_id][n]
                        tools_gpu_short_term_mem = plan_pool.tools_gpu_mem[task_id][sample_id][n]
                        tools_cpu_mem = [GlobalMetricsConfig.tools_cpu_long_term_mem[plan[i]] for i in range(0, len(plan), 2)]
                        tools_gpu_mem = [GlobalMetricsConfig.tools_gpu_long_term_mem[plan[i]] for i in range(0, len(plan), 2)]
                        if score is not None and score[0] != -2:
                            is_valid = True
                        else:
                            is_valid = False

                        token_list = [GlobalToolConfig.sop_token]
                        score_list = [0.]
                        cost_list = [0.]
                        for i in range(0, len(plan), 2):
                            token_list.append(GlobalToolConfig.tool_token_vocabulary[plan[i]])
                            score_list.append(0)

                            token_list.append(GlobalToolConfig.sod_token)
                            score_list.append(score_list[-1])

                            for dependency in plan[i + 1]:
                                token_list.append(GlobalToolConfig.dependency_token_vocabulary[dependency])
                                score_list.append(0)
                            token_list.append(GlobalToolConfig.eod_token)
                            score_list.append(score_list[-1])

                            # handle cost. we use execution price to represent execution cost.
                            idx1, idx2 = i + 2, i // 2 + 1
                            cost = calc_plan_price(tools_exec_time[:idx2], tools_cpu_mem[:idx2], tools_cpu_short_term_mem[:idx2],
                                                   tools_gpu_mem[:idx2], tools_gpu_short_term_mem[:idx2])
                            for _ in range(len(plan[i + 1]) + 3):
                                cost_list.append(cost)

                        token_list.append(GlobalToolConfig.eop_token)
                        if is_valid:
                            score_list.append(score_list[-1] + score[0])
                            cost_list.append(cost)
                        else:
                            score_list.append(GlobalMetricsConfig.score_penalty)
                            cost_list.append(GlobalMetricsConfig.cost_penalty)
                        assert len(token_list) == len(score_list) == len(cost_list)

                        self.max_score, self.min_score = max(max(score_list), self.max_score), min(min(score_list), self.min_score)
                        self.max_cost, self.min_cost = max(max(cost_list), self.max_cost), min(min(cost_list), self.min_cost)
                        self.max_plan_length = max(self.max_plan_length, len(token_list))

                        plan_dataset.append({'task_id': task_id, 'sample_id': sample_id, 'plan_tokens': token_list, 
                                        'scores': score_list, 'costs': cost_list})
        self.dataset_info.update({
            # for calculating rewards
            'max_score': self.max_score,
            'min_score': self.min_score,
            'max_cost': self.max_cost,
            'min_cost': self.min_cost,
            # to help determine the maximum size of timesteps embedding
            'max_plan_length': self.max_plan_length
        })
        return plan_dataset
    
    def _process_plan_dataset(self):
        all_states, all_actions, all_rewards = [], [], []
        for item in self.plan_dataset:
            plan_tokens, scores, costs = item['plan_tokens'], item['scores'], item['costs']
            states, actions, rewards = [], [], []
            for i in range(len(plan_tokens) - 1):
                states.append(plan_tokens[:i + 1])
                actions.append(plan_tokens[i + 1])
                rewards.append(calculate_cost_aware_reward(scores[i + 1], costs[i + 1],
                                                           self.alpha, self.min_score, self.max_score, self.min_cost, self.max_cost,
                                                           self.scale, is_real_score=(i + 1) == len(plan_tokens) - 1)) 
            self.max_reward, self.min_reward = max(max(rewards), self.max_reward), min(min(rewards), self.min_reward)

            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            
        self.dataset_info.update({
            # for normalizing rewards
            'max_reward': self.max_reward,
            'min_reward': self.min_reward,
        })

        return all_states, all_actions, all_rewards
    
    def _normalize_rewards(self):
        for rewards in self.all_rewards:
            for i in range(len(rewards)):
                rewards[i] = (rewards[i] - self.min_reward) / (self.max_reward - self.min_reward)

    def _compute_returns(self):
        """
        Compute returns (discounted cumulative rewards)
        """
        all_returns, all_timesteps = [], []
        for rewards in self.all_rewards:
            timesteps = list(range(len(rewards)))
            returns = discount_returns(rewards, self.gamma, self.scale)
            assert len(timesteps) == len(returns)
            self.max_timestep, self.min_timestep = max(max(timesteps), self.max_timestep), min(min(timesteps), self.min_timestep)
            self.max_return, self.min_return = max(returns[0], self.max_return), min(returns[0], self.min_return)
            all_timesteps.append(timesteps)
            all_returns.append(returns)
        self.dataset_info.update({
            'max_return': self.max_return,
            'min_return': self.min_return,
            'max_timestep': self.max_timestep,
            'min_timestep': self.min_timestep,
        })
        return all_returns, all_timesteps
                
    def __len__(self):
        return self.plan_num
    
    def __getitem__(self, index):
        return self.all_states[index], self.all_returns[index], self.all_actions[index], self.all_timesteps[index],\
               {'task_id': self.plan_dataset[index]['task_id'], 'sample_id': self.plan_dataset[index]['sample_id']}
    