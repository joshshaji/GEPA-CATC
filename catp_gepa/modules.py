import logging
import json
from typing import List, Union
import dspy

logger = logging.getLogger(__name__)


class GeneratePlan(dspy.Signature):
    task_query: str = dspy.InputField()
    tool_catalog_json: str = dspy.InputField()
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

    def forward(self, task_query: str, tool_catalog_json: str):
        return self.predict(task_query=task_query, tool_catalog_json=tool_catalog_json)