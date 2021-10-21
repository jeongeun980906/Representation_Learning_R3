import torch
import os
import json
import matplotlib.pyplot as plt
from utils.dataloader import torcs_dataset
from utils.plot import plot_tsne
from core.net import NONAME_model
from core.mixture_network import Mixture_model
from core.loss import mdn_sample
import torch.nn.functional as F

class test_class():
    def __init__(self,id):
        self.path = './res/{}/'.format(id)
        self.config=self.load_config(id)
        self.device = 'cuda'
        self.s_dim=29
        self.a_dim=2
        self.load_model()

    def load_config(self,id):
        with open(self.path+'config.json') as jf:
            config = json.load(jf)
        return config

    def load_model(self):
        if self.config['policy'] == 'mlp':
            self.model = NONAME_model(x_dim=int(self.config['num_traj']*31)).to(self.device)
        else:
            self.model = Mixture_model(x_dim=int(self.config['num_traj']*31)).to(self.device)
        state_dict = torch.load(self.path+'model_final.pt')
        self.model.load_state_dict(state_dict)
    
    def test(self):
        dataset = torcs_dataset(num_traj=self.config['num_traj'],train=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.model.eval()
        latent = []
        label = []
        e_loss = []
        f_loss = []
        for expert_traj,fail_traj in dataloader:
            with torch.no_grad():
                e_pred, ez = self.model(expert_traj.to(self.device))
                e_action = expert_traj[:,self.s_dim+1:self.s_dim+self.a_dim+1].to(self.device)
                f_pred, fz = self.model(fail_traj.to(self.device))
                f_action = fail_traj[:,self.s_dim+1:self.s_dim+self.a_dim+1].to(self.device)
            if self.config['policy'] == 'mdn':
                e_pred = mdn_sample(e_pred)
                f_pred = mdn_sample(f_pred)
            e_test_mse_loss = F.mse_loss(e_pred,e_action)
            f_test_mse_loss = F.mse_loss(f_pred,f_action)
            e_loss.append(e_test_mse_loss.cpu().item())
            f_loss.append(f_test_mse_loss.cpu().item())
            latent.append(ez.squeeze(0).cpu().numpy().tolist())
            label.append(0)
            latent.append(fz.squeeze(0).cpu().numpy().tolist())
            label.append(1)
        plot_tsne(latent,label,self.path,self.config)
        total_loss = f_loss+e_loss
        mean_total= sum(total_loss)/len(total_loss)
        mean_e = sum(e_loss)/len(e_loss)
        mean_f = sum(f_loss)/len(f_loss)
        strtemp = ("Average error: %.3f, expert error: %.3f, negative error: %.3f")%(mean_total,mean_e,mean_f)
        print(strtemp)