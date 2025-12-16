import torch 
import torch.nn as nn
from src.gpt_blocks.tranformer_block import TransformerBlock
from src.gpt_blocks.layer_norm import LayerNorm



class GPTModel(nn.Module):
    """ Final GPT-2 style model 124M params"""
    def __init__(self, config:dict):
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
        # preprocessing
        token_embeddings = self.token_emb(in_idx)
        position_embeddings = self.pos_embed(
            torch.arange(seq_length, device=in_idx.device)
        )
        x = self.drop_embed(token_embeddings + position_embeddings)
        
        # transformer blocks
        x = self.trf_blocks(x)
        # final normalization and output head
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
        # Logits will be of shape (batch_size, seq_length, vocab_size) 
    