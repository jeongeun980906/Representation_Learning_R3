import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utils.dataloader import torcs_dataset,synthetic_example
from utils.utils import print_n_txt
from core.net import Decoder,Policy,Encoder
from core.transformer.encoder import Transformer_Encoder
from core.loss import INFONCE_loss,BT_loss,mdn_loss,recon_loss

import torch.nn.functional as F

class SOLVER():
    def __init__(self,args):
        self.device = 'cuda'
        self.args= args
        self.state_only=True
        self.path = './res/{}/'.format(args.id)
        try:
            os.mkdir(self.path)
        except:
            pass
        self.lr = args.lr
        self.load_dataset()
        self.load_model()

    def load_dataset(self):
        if self.args.data == 'torcs':
            self.state_samples = 50
            dataset = torcs_dataset(num_traj=self.args.num_traj,state_samples=self.state_samples)
            self.traj_dim = self.args.num_traj*31
            self.a_dim = 2
            self.s_dim = 29
            self.dim = 31
            self.z_dim = 5
            self.update_samples = 5
        elif self.args.data == 'syn':
            self.state_samples = 40
            dataset = synthetic_example(num_traj=self.args.num_traj,fixed_len=self.args.fixed_len,
                            state_samples=self.state_samples)
            self.a_dim = 2
            self.s_dim = 2
            self.z_dim = 4
            if self.state_only:
                self.dim = 2
            else:
                self.dim=4
            self.traj_dim = self.args.num_traj*self.dim
            self.update_samples = 5
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,shuffle=True)

    def load_model(self):
        if self.args.encoder_base == 'mlp':
            self.encoder = Encoder(x_dim=int(self.args.num_traj*self.dim),z_dim=self.z_dim).to(self.device)
        elif self.args.encoder_base == 'transformer':
            self.encoder = Transformer_Encoder(x_dim=self.dim,z_dim=self.z_dim).to(self.device)
        self.policy = Policy(s_dim=self.s_dim,a_dim=self.a_dim,z_dim=self.z_dim).to(self.device)
        # self.model = MODEL(self.args,x_dim=int(self.args.num_traj*self.dim),s_dim=self.s_dim,a_dim=self.a_dim,
        #                         z_dim=self.z_dim,k=5,sig_max=None).to(self.device)
        if self.args.policy == 'mlp':
            self.weight = [1,1]
        elif self.args.policy == 'mdn':
            self.weight = [1,0.01]
        self.policy.init_param()
        self.encoder.init_param()
        if self.args.recon_loss:
            self.decoder = Decoder(x_dim=int(self.args.num_traj*self.dim),z_dim=5).to(self.device)

    def train(self):
        with open(self.path+'config.json','w') as jf:
            json.dump(self.args.__dict__, jf, indent=2)
        f =  open(self.path+'logs.txt', 'w') 
        self.loss = []
        self.ploss = []
        self.eloss = []
        if self.args.loss=='simclr':
            ecriterion = INFONCE_loss
        elif self.args.loss == 'BT':
            ecriterion = BT_loss
        if self.args.policy == 'mlp':
            pcriterion = F.mse_loss
        elif self.args.policy == 'mdn':
            pcriterion = mdn_loss
        flag=0
        #optimizer = torch.optim.Adam(self.model.parameters(), self.lr,weight_decay=1e-4)
        poptimizer = torch.optim.Adam(self.policy.parameters(), self.lr,weight_decay=1e-4)
        eoptimizer = torch.optim.Adam(self.encoder.parameters(), self.lr,weight_decay=1e-4)
        for e in range(200):
            total_loss = 0
            ploss = 0
            eloss = 0
            for traj_1,traj_2,state_1,action_1,state_2,action_2,_,traj_len in self.dataloader:
                if self.args.encoder_base == 'mlp':
                    traj_1 = traj_1.view(-1,self.traj_dim)
                    traj_2 = traj_2.view(-1,self.traj_dim)
                    sampled_z_1 = self.encoder(traj_1.to(self.device))
                    sampled_z_2 = self.encoder(traj_2.to(self.device))
                else:
                    traj_1 = traj_1.view(-1,self.args.num_traj,self.dim)
                    traj_2 = traj_2.view(-1,self.args.num_traj,self.dim)
                    # traj_1 = torch.cat((torch.zeros(traj_1.size(0),1,self.dim),traj_1),dim=1)
                    # traj_2 = torch.cat((torch.zeros(traj_1.size(0),1,self.dim),traj_2),dim=1)
                    sampled_z_1 = self.encoder(traj_1.to(self.device))#[:,0,:]
                    sampled_z_2 = self.encoder(traj_2.to(self.device))#[:,0,:]
                    index = traj_len.unsqueeze(-1).unsqueeze(-1).repeat(1,self.args.num_traj,self.z_dim).to(self.device)
                    sampled_z_1 = torch.gather(sampled_z_1,dim=1,index=index)[:,0,:]
                    sampled_z_2 = torch.gather(sampled_z_2,dim=1,index=index)[:,0,:]
                encoder_loss = ecriterion(sampled_z_1,sampled_z_2) # encoder loss
                if torch.sum(torch.isnan(sampled_z_1)).item()>0:
                    flag=1
                    break
                eoptimizer.zero_grad()
                encoder_loss.backward()
                eoptimizer.step()
                
                piter = self.state_samples//self.update_samples
                for p in range(piter):
                    sampled_z_1_s = sampled_z_1.detach().unsqueeze(1).repeat(1,self.update_samples,1).to(self.device)
                    sampled_z_2_s = sampled_z_2.detach().unsqueeze(1).repeat(1,self.update_samples,1).to(self.device)
                    state_1_s = state_1[:,p*self.update_samples:(p+1)*self.update_samples,:].to(self.device)
                    state_2_s = state_2[:,p*self.update_samples:(p+1)*self.update_samples,:].to(self.device)
                    action_1_s = action_1[:,p*self.update_samples:(p+1)*self.update_samples,:].to(self.device)
                    #action_1_s = action_1_s.view(-1,self.s_dim)
                    action_2_s = action_2[:,p*self.update_samples:(p+1)*self.update_samples,:].to(self.device)
                    input_1 = torch.cat((sampled_z_1_s,state_1_s),dim=-1).view(-1,self.s_dim+self.z_dim)
                    pred_1 = self.policy(input_1)
                    # print(pred_1[0,:],action_1_s[0,0,:])
                    pred_2 = self.policy(torch.cat((sampled_z_2_s,state_2_s),dim=-1).view(-1,self.s_dim+self.z_dim))
                    policy_loss = pcriterion(pred_1,action_1_s.view(-1,self.a_dim).to(self.device)) \
                                            + pcriterion(pred_2,action_2_s.view(-1,self.a_dim).to(self.device)) # Policy Loss
                    poptimizer.zero_grad()
                    policy_loss.backward()
                    poptimizer.step()
                loss = self.weight[0]*encoder_loss + self.weight[1]*policy_loss/self.update_samples
                if self.args.recon_loss:
                    recon_1 = self.decoder(sampled_z_1)
                    recon_2 = self.decoder(sampled_z_2)
                    loss += 0.1*(recon_loss(traj_1,recon_1) + recon_loss(traj_2,recon_2))
                total_loss += loss
                eloss += encoder_loss
                ploss += policy_loss/self.update_samples
            if flag==1:
                break
            total_loss /= len(self.dataloader)
            eloss /= len(self.dataloader)
            ploss /= len(self.dataloader)
            strtemp = ("Epoch: %d loss: %.3f encoder loss: %.3f policy loss: %.3f")%(e,total_loss,eloss,ploss)
            print_n_txt(_chars=strtemp, _f=f)
            self.loss.append(total_loss)
            self.eloss.append(eloss)
            self.ploss.append(ploss)
        torch.save(self.policy.state_dict(),self.path+'policy_final.pt')
        torch.save(self.encoder.state_dict(),self.path+'encoder_final.pt')

    def plot_loss(self):
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1)
        plt.title("Total Loss")
        plt.xlabel("Epochs")
        plt.plot(self.loss)

        plt.subplot(1,3,2)
        plt.title("Encoder Loss")
        plt.xlabel("Epochs")
        plt.plot(self.eloss)

        plt.subplot(1,3,3)
        plt.title("Policy Loss")
        plt.xlabel("Epochs")
        plt.plot(self.ploss)

        plt.savefig(self.path+"learning_curve.png")