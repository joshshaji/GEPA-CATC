import logging
import json
from typing import List, Union
import dspy

logger = logging.getLogger(__name__)


class GeneratePlan(dspy.Signature):
    """
    You are a cost-aware tool planning policy. Given the task query, an augmented tool catalog, and input attributes, select tools and dependencies to form an efficient plan.

    Use the augmented catalog fields to reason about costs per input size level:
    - Each tool includes cost attributes per level in `price_estimates`/`cost_attributes`.
    - The current sample’s level is already encoded via `cost_importance` and `cost_features`.
    - `weighted_cost` is the importance-weighted cost to prefer at this input level.
    - Consider `input_type`, `output_type`, and `dependencies` for compatibility.

    Objective: Maximize task performance while minimizing execution cost. Prefer plans with lower `weighted_cost` while remaining functionally valid and sufficient for the task. Output format must strictly alternate tool then its dependency list.
    """
    task_query: str = dspy.InputField()
    tool_catalog_json: str = dspy.InputField()
    input_attributes_json: str = dspy.InputField()
    plan_json: List[Union[str, List[str]]] = dspy.OutputField(
        desc=(
            "A sequential list representing a plan of tool calls. "
            "It must strictly alternate between two types of elements: "
            "1. A tool name (a string, e.g., 'image_denoising'). "
            "2. Its direct dependency (a list containing exactly one string, "
            "e.g., ['input_of_query'] or ['output_of_previous_tool']). "
            "The overall list must contain an even number of elements."
        )
    )

class PlanGenerator(dspy.Module):
    def __init__(self, **predict_kwargs):
        super().__init__()
        self.predict = dspy.Predict(GeneratePlan, **predict_kwargs)

    def forward(self, task_query: str, tool_catalog_json: str, input_attributes_json: str):
        return self.predict(task_query=task_query, tool_catalog_json=tool_catalog_json, input_attributes_json=input_attributes_json)
