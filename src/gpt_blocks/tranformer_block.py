import torch 
import torch.nn as nn
from src.gpt_blocks.self_attention import MultiHeadAttention
from src.gpt_blocks.nueral_network import FeedForward
from src.gpt_blocks.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg:dict):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["embed_dim"],
                                      d_out=cfg["embed_dim"],
                                      context_length=cfg["context_length"],
                                      num_heads=cfg["n_heads"],
                                      dropout=cfg["drop_rate"],
                                      qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut
        return x