"""
Define tool tokens and dependency tokens.
"""
import torch.nn as nn
from src.config import GlobalToolConfig


class ToolTokens(nn.Module):
    def __init__(self, num_tools, embed_dim):
        super().__init__()
        self.num_tools = num_tools
        self.embed_dim = embed_dim
        # number of entries = num_tools + [SOP] (Start of Plan) + [EOP] (End of Plan) + <unknown>
        self.tool_embeddings = nn.Embedding(num_tools + 3, embed_dim)  
    
    def forward(self, x):
        return self.tool_embeddings(x - GlobalToolConfig.tool_token_start)


class DependencyTokens(nn.Module):
    def __init__(self, num_dependency, embed_dim):
        super().__init__()
        self.num_dependency = num_dependency
        self.embed_dim = embed_dim
        # number of entries = num_dependency + <unknown>
        self.dependency_embeddings = nn.Embedding(num_dependency + 1, embed_dim)  
    
    def forward(self, x):
        return self.dependency_embeddings(x - GlobalToolConfig.dependency_token_start)

