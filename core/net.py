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
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

'''
VAE Decoder
''' 
class Decoder(nn.Module):
    def __init__(self, x_dim=93, h_dim=32, z_dim=5):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
        self.mu_min = -3
        self.mu_max = 3
        
    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mixture_head(h)

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        """
        Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        """
        self.policy.mixture_head.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)

class MixtureEncoder(nn.Module):
    def __init__(self,x_dim=93, h_dim=32, z_dim=5,k=5,sig_max=None):
        super(MixtureEncoder,self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.mixture_head = MixturesOfGaussianLayer(h_dim,z_dim,k,sig_max)
        self.mu_min = -3
        self.mu_max = 3

    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mixture_head(h)

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        """
        Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        """
        self.mixture_head.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)



class MODEL(nn.Module):
    def __init__(self,args,test=False,x_dim=93,s_dim=29,a_dim=2,h_dim=32,z_dim=5,k=5,sig_max=None):
        super(MODEL,self).__init__()
        self.s_dim = 29
        if test:
            if args['encoder'] == 'mdn':
                self.encoder = MixtureEncoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim,k=k,sig_max=sig_max)
            else:
                self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
            if args['policy'] == 'mdn':
                self.policy = MixturePolicy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,z_dim=z_dim,k=k,sig_max=sig_max)
            else:
                self.policy = Policy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,z_dim=z_dim)
        else:
            if args.encoder == 'mdn':
                self.encoder = MixtureEncoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim,k=k,sig_max=sig_max)
            else:
                self.encoder = Encoder(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
            if args.policy == 'mdn':
                self.policy = MixturePolicy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,z_dim=z_dim,k=k,sig_max=sig_max)
            else:
                self.policy = Policy(s_dim=s_dim,a_dim=a_dim,h_dim=h_dim,z_dim=z_dim)

    def forward(self,x,state):
        z = self.encoder(x)
        inputs = torch.cat((state,z),dim=1)
        pred = self.policy(inputs)
        return pred,z

    def init_param(self):
        self.encoder.init_param()
        self.policy.init_param()