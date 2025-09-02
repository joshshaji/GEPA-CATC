"""
Provide some functionalities related to costs (e.g., calculating plan execution prices).

"""

from bisect import bisect_left
from src.config import GlobalMetricsConfig, GlobalDataConfig, GlobalToolConfig


CPU_LONG_MEMORY_LEVELS = list(sorted(GlobalMetricsConfig.cpu_long_memory_pricing.keys()))
GPU_LONG_MEMORY_LEVELS = list(sorted(GlobalMetricsConfig.gpu_long_memory_pricing.keys()))


def cacl_tool_price(exec_time, cpu_long_term_mem, cpu_short_term_mem, gpu_long_term_mem, gpu_short_term_mem):
    """
    The tool execution price is calculated by:
    Price = exec_time x (cpu_long_term_mem x cpu_long_term_mem_pricing 
                    + cpu_short_term_mem x cpu_short_term_mem_pricing) 
                    + (gpu_long_term_mem x gpu_long_term_mem_pricing 
                    + gpu_short_term_mem x gpu_short_term_mem_pricing) 
          + price_per_request
    """
    level_cpu = min(bisect_left(CPU_LONG_MEMORY_LEVELS, cpu_long_term_mem), len(CPU_LONG_MEMORY_LEVELS) - 1)
    level_cpu = CPU_LONG_MEMORY_LEVELS[level_cpu]
    level_gpu = min(bisect_left(GPU_LONG_MEMORY_LEVELS, gpu_long_term_mem), len(GPU_LONG_MEMORY_LEVELS) - 1)
    level_gpu = GPU_LONG_MEMORY_LEVELS[level_gpu]
    return GlobalMetricsConfig.price_per_request + exec_time * (cpu_long_term_mem * GlobalMetricsConfig.cpu_long_memory_pricing[level_cpu]
                                                 + cpu_short_term_mem * GlobalMetricsConfig.cpu_short_memory_pricing_per_mb
                                                 + gpu_long_term_mem * GlobalMetricsConfig.gpu_long_memory_pricing[level_gpu]
                                                 + gpu_short_term_mem * GlobalMetricsConfig.gpu_short_memory_pricing_per_mb)


def calc_plan_price(tools_exec_time, tools_cpu_long_term_mem, tools_cpu_short_term_mem, tools_gpu_long_term_mem, 
                    tools_gpu_short_term_mem):
    """
    The price of a plan is calculated by the sum of price of each tool in the plan.
    """
    assert type(tools_exec_time) is list, "please place the cost information of each tool in a list."
    assert len(tools_exec_time) == len(tools_cpu_long_term_mem) == len(tools_cpu_short_term_mem) == len(tools_gpu_long_term_mem) \
           == len(tools_gpu_short_term_mem)
    plan_price = 0.
    for i in range(len(tools_exec_time)):
        plan_price += cacl_tool_price(tools_exec_time[i], tools_cpu_long_term_mem[i], tools_cpu_short_term_mem[i],
                                      tools_gpu_long_term_mem[i], tools_gpu_short_term_mem[i])
    return plan_price


def determine_input_level(input_info):
    if input_info['has_image']:
        image_size_levels = GlobalDataConfig.image_sizes
        image_size = input_info['image_size'][0] * input_info['image_size'][1]
        # print(image_size)
        # search for the input size level that matches the input size 
        size_level = bisect_left(image_size_levels, image_size)
        if size_level >= len(image_size_levels):
            size_level -= 1
        elif size_level != 0 and image_size_levels[size_level] - image_size > abs(image_size_levels[size_level - 1] - image_size):
            size_level -= 1
        return size_level
    if input_info['has_text']:
        text_length_levels = GlobalDataConfig.text_lengths
        text_length = input_info['text_length']
        # search for the input size level that matches the input size 
        size_level = bisect_left(text_length_levels, text_length)
        if size_level >= len(text_length_levels):
            size_level -= 1
        elif size_level != 0 and text_length_levels[size_level] - text_length > abs(text_length_levels[size_level - 1] - text_length):
            size_level -= 1
        return size_level
    return None

def estimate_tool_price(tool, input_size):
    """
    Estimate the execution price of the tool.
    Used when LLM generate tool plans.
    """
    return GlobalToolConfig.tool_prices[tool][input_size]
