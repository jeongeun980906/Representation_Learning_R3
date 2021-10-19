import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as TD
from torch.autograd import Variable
import numpy as np

'''
VAE encoder
''' 
class Encoder(nn.Module):
    def __init__(self, x_dim=93, h_dim=32, z_dim=5):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, z_dim)
        self.fc32 = nn.Linear(h_dim, z_dim)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu =  self.fc31(h)
        log_var = self.fc32(h)
        return self.sampling(mu, log_var)

'''
MLP policy
'''

class Policy(nn.Module):
    def __init__(self,s_dim=29,a_dim=2,h_dim=32,z_dim=5):
        super(Policy,self).__init__()

        self.fc1 = nn.Linear(s_dim+z_dim,h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim)
        self.fc3 = nn.Linear(h_dim,a_dim)

    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)