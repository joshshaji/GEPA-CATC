from collections import defaultdict
import pickle


class PlanPool:
    """
    A class that records the information of the generated plan for each data sample.
    We organize the information as follows:
    {
        task_id1: 
            {
                sample_id1: plans/scores/costs
                ...
            }
        ...
    }

    Author: Duo Wu
    """
    def __init__(self):
        self.plans = defaultdict(dict)
        self.scores = defaultdict(dict)
        self.tools_exec_time = defaultdict(dict)
        self.tools_cpu_mem = defaultdict(dict)
        self.tools_gpu_mem = defaultdict(dict)
    
    def batch_add(self, task_id, sample_ids, plans, scores, costs, is_module_seq=True):
        for i in range(len(sample_ids)):
            self.add_sample(task_id, sample_ids[i], plans[i], scores[i], costs[i], is_module_seq=is_module_seq)
    
    def add_sample(self, task_id, sample_id, plan, score, cost, is_module_seq=True):
        if is_module_seq:
            # if the plan is represented as a module sequence ([tool1, tool2, tool3, ...])
            # transform it into the form that is more like our tool planning language:
            # [tool1, [tool1_depedency], tool2, [tool2_dependency], ...]
            plan = plan.split(', ')
            new_plan = []
            for i in range(len(plan)):
                new_plan.append(plan[i])
                if i == 0:
                    new_plan.append(['Input of Query'])
                else:
                    new_plan.append([plan[i - 1]])
            plan = new_plan

        self.plans[task_id][sample_id] = plan
        self.scores[task_id][sample_id] = score
        if cost is None:
            self.tools_exec_time[task_id][sample_id] = None
            self.tools_cpu_mem[task_id][sample_id] = None
            self.tools_gpu_mem[task_id][sample_id] = None
        else:
            self.tools_exec_time[task_id][sample_id] = cost[0]
            self.tools_cpu_mem[task_id][sample_id] = cost[1]
            self.tools_gpu_mem[task_id][sample_id] = cost[2]

    def print_info(self):
        for task_id, samples in self.plans.items():
            print('=' * 30)
            print('Task id:', task_id)
            for sample_id in sorted(samples.keys()):
                print('    Sample id:', sample_id)
                print('        Plans:', self.plans[task_id][sample_id])
                print('        Scores:', self.scores[task_id][sample_id])
                print('        Tools Execution time:', self.tools_exec_time[task_id][sample_id])
                print('        Tools CPU Memory:', self.tools_cpu_mem[task_id][sample_id])
                print('        Tools GPU Memory:', self.tools_gpu_mem[task_id][sample_id])
            print('=' * 30)

def load_plan_tool(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data
 
    