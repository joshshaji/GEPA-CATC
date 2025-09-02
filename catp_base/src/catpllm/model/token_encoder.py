import torch
import torch.nn as nn

from src.catpllm.model.tokens import ToolTokens, DependencyTokens
from src.config import GlobalToolConfig


class TokenEncoder(nn.Module):
    """
    The encoder network for encoding tool/dependency tokens.
    """
    def __init__(self, num_tokens, embed_dim, cost_dim=4, num_heads=8, device='cpu'):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.cost_dim = cost_dim
        self.device = device
        self.tool_tokens = ToolTokens(num_tokens, embed_dim)
        self.dependency_tokens = DependencyTokens(num_tokens, embed_dim)
        self.linear = nn.Linear(1, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)

    def forward(self, tokens, sample_size, fuse_cost=True):
        token_embeddings = []
        for token in tokens:
            if token < GlobalToolConfig.dependency_token_start:  # tool token
                token_embedding = self.tool_tokens(token)
                if token != GlobalToolConfig.sop_token and token != GlobalToolConfig.eop_token and fuse_cost:
                    # fuse cost features into token embeddings
                    cost_attr = torch.tensor(GlobalToolConfig.tool_prices[GlobalToolConfig.tool_token_vocabulary_reverse[token.item()]], dtype=torch.float32, 
                                             device=self.device).reshape(-1, 1)
                    # use cosine vector to indicate the current sample size
                    cosine_vector = self._create_cosine_vector(sample_size)
                    cost_features = self.linear(cost_attr) + cosine_vector
                    self_attn_inputs = torch.cat([cost_features, token_embedding.unsqueeze(0)], dim=0)
                    self_attn_mask = nn.Transformer.generate_square_subsequent_mask(self_attn_inputs.shape[0], device=self.device)
                    self_attn_outputs = self.self_attn(self_attn_inputs, self_attn_inputs, self_attn_inputs, attn_mask=self_attn_mask, 
                                                       is_causal=True)
                    token_embedding = self_attn_outputs[0][-1]
                    pass
            else:  # dependency token
                token_embedding = self.dependency_tokens(token)
            token_embeddings.append(token_embedding)
        token_embeddings = torch.stack(token_embeddings)
        return token_embeddings

    def _create_cosine_vector(self, sample_size):
        """
        Create a cosine vector according to the current sample size, which is always centered at the current sample size.
        For instance, for sample size = 2 (indexed from 0) and number of sample sizes = 4: 
        cosine vector = [0.5000, 0.8660, 1.0000, 0.8660])
        This allows the importance of cost features at different sample sizes gradually decreases according to their distance
        between the current sample size.
        """
        sample_sizes = torch.arange(0, self.cost_dim, device=self.device).reshape(-1, 1)
        sample_sizes = (sample_sizes - sample_size) / (self.cost_dim - 1)
        cosine_vector = torch.cos(sample_sizes * torch.pi / 2)
        return cosine_vector
    