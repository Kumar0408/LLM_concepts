import torch 
import torch.nn as nn

class GELU(nn.Module):
    # Gaussian Error Linear Unit activation function
    # Obtained from curve fiiting to the actual GELU function as in GPT2 model
    def __init__(self):
        super().__init__()

    def forward(self, x):
                return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
                ))