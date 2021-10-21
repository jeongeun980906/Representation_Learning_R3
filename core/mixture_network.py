import torch.nn as nn
from collections import OrderedDict
from core.net import Encoder
import torch.nn.functional as F
import torch

class MixturesOfGaussianLayer(nn.Module):
    def __init__(self,in_dim,y_dim,k,sig_max=None):
        super(MixturesOfGaussianLayer,self).__init__()
        self.in_dim = in_dim
        self.y_dim = y_dim
        self.k = k
        self.sig_max = sig_max

        self.fc_pi = nn.Linear(self.in_dim,self.k)
        self.fc_mu = nn.Linear(self.in_dim,self.k*self.y_dim)
        self.fc_sigma = nn.Linear(self.in_dim,self.k*self.y_dim)

    def forward(self,x):
        pi_logit = self.fc_pi(x) # [N x K]
        pi = torch.softmax(pi_logit,1) # [N x K]
        mu = self.fc_mu(x) # [N x KD]
        mu = torch.reshape(mu,(-1,self.k,self.y_dim)) # [N x K x D]
        sigma = self.fc_sigma(x) # [N x KD]
        sigma = torch.reshape(sigma,(-1,self.k,self.y_dim)) # [N x K x D]
        if self.sig_max is None:
            sigma = torch.exp(sigma) # [N x K x D]
        else:
            sigma = self.sig_max * (torch.sigmoid(sigma) + 1e-8) # [N x K x D]
        out = {'pi':pi,'mu':mu,'sigma':sigma}
        return out

class MixturePolicy(nn.Module):
    def __init__(self,s_dim=29,a_dim=2,h_dim=32,z_dim=5,k=5,sig_max=None):
        super(MixturePolicy,self).__init__()
        self.fc1 = nn.Linear(s_dim+z_dim,h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim)
        self.mixture_head = MixturesOfGaussianLayer(h_dim,a_dim,k,sig_max)

    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mixture_head(h)

class Mixture_model(nn.Module):
    def __init__(self,x_dim=93,s_dim=29,
                a_dim=2,h_dim=32,z_dim=5,k=5,sig_max=None):
        super(Mixture_model,self).__init__()
        self.s_dim = 29
        self.mu_min = -3
        self.mu_max = 3
        self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        self.policy = MixturePolicy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,
                                z_dim=z_dim,k=k,sig_max=sig_max)
    def forward(self,x):
        z = self.encoder(x)
        state = x[:,:self.s_dim]
        inputs = torch.cat((state,z),dim=1)
        pred = self.policy(inputs)
        return pred,z

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        """
        Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        """
        self.policy.mixture_head.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)
