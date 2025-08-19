from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    cost_attr

class CATPDataset(BaseModel):
    query: str
    tools: list[Tool]