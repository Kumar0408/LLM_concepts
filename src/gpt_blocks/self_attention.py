import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    # Simple aelf attention mechanism
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        # x shape: (seq_len, d_in)
        queries = x @ self.W_query  # shape: (seq_len, d_out)
        keys = x @ self.W_key       # shape: (seq_len, d_out)
        values = x @ self.W_value   # shape: (seq_len, d_out)

        d_k = keys.shape[-1]
        attention_scores = queries@keys.T  # shape: (seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores/d_k**0.5, dim=-1)  # shape: (seq_len, seq_len)

        context_vec = attention_weights @ values  # shape: (seq_len, d_out)
        return context_vec

class SelfAttention_v2(nn.Module):
    # Self attention with trainable Q,K and V vectors
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        # x shape: (seq_len, d_in)
        queries =  self.W_query(x)  
        keys = self.W_key(x)      
        values =  self.W_value(x)  

        d_k = keys.shape[-1]
        attention_scores = queries@keys.T  # shape: (seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores/d_k**0.5, dim=-1)  # shape: (seq_len, seq_len)

        context_vec = attention_weights @ values  # shape: (seq_len, d_out)
        return context_vec

class CausalAttention(nn.Module):
    # Self attention with Masking and Dropout
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    
    def forward(self,x):
        b, num_tokens,d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries@keys.transpose(1,2) #exclude batch shape
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5,dim = -1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights@values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    # Wrapper to create multiple heads of attention and concatenate their outputs
    def __init__(self, d_in, d_out, context_length,
    dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
        [CausalAttention(
        d_in, d_out, context_length, dropout, qkv_bias)
        for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    # Multi-head self attention with masking and dropout with mixing of heads
    def __init__(self, 
                 d_in,
                 d_out,
                 context_length,
                 num_heads,
                 dropout,
                 qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Final output projection - mixes the information from different heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Linear projections
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)       # (b, num_tokens, d_out)
        values = self.W_value(x)   # (b, num_tokens, d_out)

        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)        # (b, num_heads, num_tokens, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)    # (b, num_heads, num_tokens, head_dim)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2,3) 
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_heads, num_tokens, head_dim)
        context_vec = context_vec.transpose(1, 2) # (b, num_tokens, num_heads, head_dim) transpose back
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)

        output_context_vec = self.out_proj(context_vec)  # Final linear projectio
        return output_context_vec