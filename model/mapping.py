import torch
import torch.nn.functional as F
from torch import nn

class IM2TEXT(nn.Module):       #mapping network
    def __init__(self, embed_dim=512, middle_dim=768, output_dim=512, n_layer=1, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())            
            dim = middle_dim
            layers.append(nn.Sequential(*block))        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)