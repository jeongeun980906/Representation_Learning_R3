import math
from numpy import NaN
import torch
from torch.autograd import Variable
import torch.distributions as TD

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
T=2
device = 'cuda'

def l1(i,j,s,N):
    non = torch.sum(torch.exp(s[i,:N]/T))-torch.exp(s[i,i]/T)
    return 100*torch.exp(s[i,j]/T)/non

def l2(i,j,s,N):
    non = torch.sum(torch.exp(s[i,N:]/T))-torch.exp(s[i,i]/T)
    return 100*torch.exp(s[i,j]/T)/non

def simclr_loss(u,v):
    '''
    expert u [N x z]
    fail v [N x z]
    '''
    N = v.size(0)
    z = torch.cat((u,v))
    # print(z)
    z = torch.nn.functional.normalize(z, dim=-1)
    # print(z)
    s = torch.matmul(z, z.T) # [2N x 2N]
    loss = 0
    for k in range(N):
        loss+= l1(k,N+k,s,N) + l2(N+k,k,s,N)
    loss /= (2*N)
    return loss

def BT_loss(u,v):
    '''
    expert u [N x z]
    fail v [N x z]
    '''
    u = torch.nn.functional.normalize(u,dim=1).unsqueeze(-1)
    v = torch.nn.functional.normalize(v,dim=1)
    v = torch.reshape(v,(v.size(0),1,-1))
    C = torch.bmm(u,v)
    C = torch.mean(C,dim=0) # [z x z]
    mask = torch.eye(C.size(0)).to('cuda')
    loss = mask*torch.square(C) + 0.01*torch.square(1-C)*(torch.ones_like(mask)-mask)
    loss = torch.sum(loss)
    return loss

def mdn_loss(out,data):
    """
    pi: [N x K]
    mu: [N x K x D]
    sigma: [N x K x D]
    data: [N x D]
    """
    pi = out['pi']
    mu = out['mu']
    sigma = out['sigma']
    data_usq = torch.unsqueeze(data,1) # [N x 1 x D]
    data_exp = data_usq.expand_as(sigma) # [N x K x D]
    probs = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data_exp-mu)/sigma)**2) / sigma # [N x K x D]
    probs_prod = torch.prod(probs,2) # [N x K]
    prob = torch.sum(probs_prod*pi,dim=1) # [N]
    prob = torch.clamp(prob,min=1e-8) # Clamp if the prob is to small
    nll = -torch.log(prob) # [N] 
    # out = {'data_usq':data_usq,'data_exp':data_exp,
    #        'probs':probs,'probs_prod':probs_prod,'prob':prob,'nll':nll}
    nll = torch.mean(nll)
    return nll

def mdn_uncertainties(out):
    # Compute Epistemic Uncertainty
    pi = out['pi']
    mu = out['mu']
    sigma = out['sigma']
    M = 0.1# 0.1
    pi = torch.softmax(M*pi,1) # (optional) heuristics 
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]
    mu_avg = torch.sum(torch.mul(pi_exp,mu),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_exp = mu_avg.expand_as(mu) # [N x K x D]
    mu_diff_sq = torch.square(mu-mu_exp) # [N x K x D]
    epis_unct = torch.sum(torch.mul(pi_exp,mu_diff_sq), dim=1)  # [N x D]

    # Compute Aleatoric Uncertainty
    alea_unct = torch.sum(torch.mul(pi_exp,sigma), dim=1)  # [N x D]
    # Sqrt
    epis_unct,alea_unct = torch.sqrt(epis_unct),torch.sqrt(alea_unct)
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) #[N]
    out = {'epis':epis_unct,'alea':alea_unct,'pi_entropy':entropy_pi}
    return out
    
def mdn_sample(out):
    """
    pi: [N x K]
    mu: [N x K x D]
    sigma: [N x K x D]
    """
    pi = out['pi']
    mu = out['mu']
    sigma = out['sigma']
    categorical = TD.Categorical(pi)
    mixture_list = list(categorical.sample().data)
    _N,_D = sigma.size(0),sigma.size(2)
    eps = Variable(torch.empty(_N,_D).normal_()).to(device) # [N x D]
    sample = torch.empty_like(eps) # [N x D]
    for i_idx, mixture_idx in enumerate(mixture_list):
        mu_i,sigma_i = mu[i_idx,mixture_idx],sigma[i_idx,mixture_idx]
        sample[i_idx] = eps[i_idx].mul(sigma_i).add(mu_i)
    return sample # [N x D]