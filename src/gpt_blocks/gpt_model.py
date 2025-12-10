import torch 
import torch.nn as nn
from src.gpt_blocks.self_attention import MultiHeadAttention
from src.gpt_blocks.nueral_network import FeedForward
from src.gpt_blocks.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
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


class GPTModel(nn.Module):
    """ Final GPT-2 style model 124M params"""
    def __init__(self, config : dict):
        super().__init__()
        self.token_emb = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_embed = nn.Embedding(config['context_length'], config['embed_dim'])
        self.drop_embed = nn.Dropout(config['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config)
            for _ in range(config["n_layers"])]
        )
        
        self.final_norm = LayerNorm(config['embed_dim'])
        self.out_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_length = in_idx.size()
        token_embeddings = self.token_emb(in_idx)
        position_embeddings = self.pos_embed(
            torch.arange(seq_length, device=in_idx.device)
        )
        x = self.drop_embed(token_embeddings + position_embeddings)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


