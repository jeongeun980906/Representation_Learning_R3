import torch
T=2
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
    loss = mask*torch.square(C) + 0.1*torch.square(1-C)*(torch.ones_like(mask)-mask)
    loss = torch.sum(loss)
    return loss