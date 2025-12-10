import torch.nn as nn
import torch
from src.gpt_blocks.activation_function import GELU

class FeedForward(nn.Module):
    """ Feed forward neural network with GELU activation """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
                        GELU(),
                        nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
         )
        
    def forward(self, x):
        return self.layers(x)