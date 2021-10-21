import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return mu#self.sampling(mu, log_var)

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

class NONAME_model(nn.Module):
    def __init__(self,x_dim=93,s_dim=29,a_dim=2,h_dim=32,z_dim=5):
        super(NONAME_model,self).__init__()
        self.s_dim = 29
        self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.policy = Policy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,z_dim=z_dim)

    def forward(self,x):
        z = self.encoder(x)
        state = x[:,:self.s_dim]
        #ex_action = x[:,self.s_dim+1:self.s_dim+self.a_dim+1].to(self.device)
        inputs = torch.cat((state,z),dim=1)
        pred = self.policy(inputs)
        return pred,z
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)