from numpy import random
import torch
import os
import json
import matplotlib.pyplot as plt
from utils.dataloader import torcs_dataset,synthetic_example
from utils.plot import plot_tsne
from core.net import Policy,Encoder
from core.transformer.encoder import Transformer_Encoder
from core.loss import mdn_sample
import torch.nn.functional as F

class test_class():
    def __init__(self,id,plot_PN):
        self.path = './res/{}/'.format(id)
        self.plot_PN = plot_PN
        self.config=self.load_config(id)
        self.device = 'cuda'
        if self.config['data'] == 'torcs':
            self.dim = 31    
            self.s_dim= 29
            self.a_dim = 2
            self.z_dim=5
        elif self.config['data'] == 'syn':
            if self.config['state_only']:
                self.dim = 2
            else:
                self.dim=4
            self.s_dim = 2
            self.a_dim = 2
            self.z_dim = 4
        self.traj_dim = self.config['num_traj']*self.dim
        self.load_model()

    def load_config(self,id):
        with open(self.path+'config.json') as jf:
            config = json.load(jf)
        return config

    def load_model(self):
        self.policy = Policy(s_dim=self.s_dim,a_dim=self.a_dim,z_dim=self.z_dim).to(self.device)
        state_dict = torch.load(self.path+'policy_final.pt')
        self.policy.load_state_dict(state_dict)
        if self.config['encoder_base']== 'mlp':
            self.encoder = Encoder(x_dim=int(self.config['num_traj']*self.dim),z_dim=self.z_dim).to(self.device)
        else:
            self.encoder = Transformer_Encoder(x_dim=self.dim,z_dim=self.z_dim).to(self.device)
        state_dict = torch.load(self.path+'encoder_final.pt')
        self.encoder.load_state_dict(state_dict)

    def test(self):
        if self.config['data'] == 'torcs':
            dataset = torcs_dataset(num_traj=self.config['num_traj'],train=False)
        elif self.config['data'] == 'syn':
            dataset = synthetic_example(num_traj=self.config['num_traj'],
                            fixed_len=self.config['fixed_len'],state_only= self.config['state_only'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.policy.eval()
        self.encoder.eval()
        latent = []
        label = []
        loss= []
        for _ in range(100):
            for traj_1,traj_2,state_1,action_1,state_2,action_2,traj_type,traj_len in dataloader:
                #print(traj_len)
                with torch.no_grad():
                    if self.config['encoder_base']== 'mlp':
                        traj_1 = traj_1.view(-1,self.traj_dim).to(self.device)
                        traj_2 = traj_2.view(-1,self.traj_dim).to(self.device)
                        z_1 = self.encoder(traj_1)
                        z_2 = self.encoder(traj_2)
                    else:
                        traj_1 = traj_1.view(-1,self.config['num_traj'],self.dim)
                        traj_2 = traj_2.view(-1,self.config['num_traj'],self.dim)
                        # traj_1 = torch.cat((traj_1,torch.zeros(traj_1.size(0),1,self.dim)),dim=1)
                        # traj_2 = torch.cat((traj_2,torch.zeros(traj_1.size(0),1,self.dim)),dim=1)
                        z_1 = self.encoder(traj_1.to(self.device))#[:,0,:]
                        z_2 = self.encoder(traj_2.to(self.device))#[:,0,:]
                        z_1 = torch.mean(z_1,axis=1)
                        z_2 = torch.mean(z_2,axis=1)
                    #pred_1, z_1 = self.model(traj_1.view(-1,self.traj_dim).to(self.device),state_1[:,0,:].to(self.device))
                    #pred_2,z_2 = self.model(traj_2.view(-1,self.traj_dim).to(self.device),state_2[:,0,:].to(self.device))
                    pred_1 = self.policy(torch.cat((z_1,state_1[:,0,:].to(self.device)),dim=-1))
                    pred_2 = self.policy(torch.cat((z_2,state_2[:,0,:].to(self.device)),dim=-1))
                if self.config['policy'] == 'mdn':
                    pred_1 = mdn_sample(pred_1)
                    pred_2 = mdn_sample(pred_2)
                test_mse_loss = (F.mse_loss(pred_1,action_1[:,0,:].to(self.device))+F.mse_loss(pred_2,action_2[:,0,:].to(self.device)))/2
                if self.config['data'] == 'syn':
                    traj_type = traj_type[0,0].item()+10*traj_type[0,2].item()#+round(traj_type[0,1].item(),2)
                loss.append(test_mse_loss.cpu().item())
                latent.append(z_1.squeeze(0).cpu().numpy().tolist())
                label.append(traj_type)
                latent.append(z_2.squeeze(0).cpu().numpy().tolist())
                label.append(traj_type)
        plot_tsne(latent,label,self.path,self.config,self.plot_PN)
        mean_total= sum(loss)/len(loss)
        strtemp = ("Average error: %.3f")%(mean_total)
        print(strtemp)

    def simulate_syn(self):
        if self.config['data'] == 'torcs':
            dataset = torcs_dataset(num_traj=self.config['num_traj'],train=False)
        elif self.config['data'] == 'syn':
            dataset = synthetic_example(num_traj=self.config['num_traj'])
        #choices = random.choice(108,3)
        choices = [2,11,21]
        plt.figure(figsize=(10,10))
        for i in choices:
            traj_1,_,_,_,_,_,traj_type,traj_len = dataset.__getitem__(i)
            traj_type = traj_type.numpy().tolist()
            traj_type_label = "vel: %d type: %d"%(traj_type[0],traj_type[2])
            self.policy.eval()
            if self.config['encoder_base']== 'mlp':
                z = self.encoder(traj_1.view(-1,self.traj_dim).to(self.device)).detach()
            else:
                traj_t = traj_1.view(-1,self.config['num_traj'],self.dim)
                traj_t = traj_t = torch.cat((traj_t,torch.zeros(traj_t.size(0),1,self.dim)),dim=1)
                z = self.encoder(traj_t.to(self.device)).detach()
                #z = z[:,0,:]
                z = torch.mean(z,axis=1)
            past_traj_x = traj_1[0::self.dim].numpy().tolist()[:traj_len]
            past_traj_y = traj_1[1::self.dim].numpy().tolist()[:traj_len]
            plt.plot(past_traj_x,past_traj_y,marker='o',linewidth=1, markersize=2,label='given '+traj_type_label)
            x = past_traj_x[-1]
            y = past_traj_y[-1]
            traj_x = []
            traj_y = []
            for i in range(50):
                traj_x.append(x)
                traj_y.append(y)
                with torch.no_grad():
                    input = torch.FloatTensor([x,y]).unsqueeze(0).to(self.device)
                    input = torch.cat((z,input),dim=-1)
                    prediction = self.policy(input)
                    dx = prediction[0][0].cpu().item()
                    dy = prediction[0][1].cpu().item()
                    x += dx/10
                    y += dy/10
                    #print(dx,dy)
            z_list = z.cpu().numpy().tolist()
            plt.plot(traj_x,traj_y,marker='o',linewidth=1, markersize=2,label='predicted')
            
            # fake_z = torch.zeros_like(z)
            # x = traj_1[-4].item()
            # y =traj_1[-3].item()
            # traj_x = []
            # traj_y = []
            # for i in range(50):
            #     traj_x.append(x)
            #     traj_y.append(y)
            #     with torch.no_grad():
            #         input = torch.FloatTensor([x,y]).unsqueeze(0).to(self.device)
            #         input = torch.cat((fake_z,input),dim=-1)
            #         prediction = self.policy(input)
            #         dx = prediction[0][0].cpu().item()
            #         dy = prediction[0][1].cpu().item()
            #         x += dx/10
            #         y += dy/10
            #         #print(dx,dy)
            # z_list = z.cpu().numpy().tolist()
            # plt.plot(traj_x,traj_y,marker='o',linewidth=1, markersize=2,label='fake z')
        plt.ylim(-4,4)
        plt.xlim(-4,4)
        plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.savefig(self.path+"z.png")