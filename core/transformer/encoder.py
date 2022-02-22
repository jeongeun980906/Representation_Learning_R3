from core.transformer.common import MultiHeadAttentionLayer,Point_Wise_FeedForward,PositionalEncoding

import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
import math

class EncoderLayer(nn.Module):
    def __init__(self,dk=8, dv=8,d_model=32,n_heads=4,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.mha = MultiHeadAttentionLayer(dk=dk, dv=dv, d_model=d_model,n_heads=n_heads, dropout=dropout)
        self.pwf = Point_Wise_FeedForward(d_model = d_model,dropout=dropout)
    def forward(self,x,mask):
        query,key,value = x,x,x
        x_1,_ = self.mha(query,key,value,mask)
        # x = x + x_1
        x = self.LayerNorm(x+x_1)
        x_1 = self.pwf(x)
        # x = x + x_1
        x = self.LayerNorm(x+x_1)
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self,x_dim=4,dk=16, dv=16,d_model=64,n_heads=4,dropout=0.1, nx=4,z_dim=3):
        super(Transformer_Encoder,self).__init__()
        self.embedding = nn.Linear(x_dim,d_model)
        self.positional_encodding = PositionalEncoding(emsize=d_model,dropout=dropout)
        self.layers = nn.ModuleList([])
        for _ in range(nx):
            self.layers.append(
                EncoderLayer(dk=dk, dv=dv,d_model=d_model,n_heads=n_heads,dropout=dropout)
            )
        self.layers = nn.Sequential(*self.layers)
        self.out = nn.Linear(d_model,z_dim)
    
    def forward(self,x,mask=None):
        x = self.embedding(x)
        x = self.positional_encodding(x)
        for layer in self.layers:
            x = layer(x,mask)
        x = self.out(x)
        return x

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)