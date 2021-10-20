import torch
import os
import json
import matplotlib.pyplot as plt
from utils.dataloader import torcs_dataset
from utils.utils import print_n_txt
from core.net import NONAME_model
from core.loss import simclr_loss,BT_loss

import torch.nn.functional as F

class SOLVER():
    def __init__(self,args):
        self.device = 'cuda'
        self.args= args
        self.path = './res/{}/'.format(args.id)
        try:
            os.mkdir(self.path)
        except:
            pass
        self.lr = args.lr
        self.load_dataset()
        self.load_model()

    def load_dataset(self):
        dataset = torcs_dataset(num_traj=self.args.num_traj)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
        self.traj_dim = 3*31
        self.a_dim = 2
        self.s_dim = 29

    def load_model(self):
        self.model = NONAME_model(x_dim=int(self.args.num_traj*31)).to(self.device)
        self.model.init_param()

    def train(self):
        with open(self.path+'config.json','w') as jf:
            json.dump(self.args.__dict__, jf, indent=2)
        f =  open(self.path+'logs.txt', 'w') 
        self.loss = []
        self.ploss = []
        self.eloss = []
        if self.args.loss=='simclr':
            criterion = simclr_loss
        elif self.args.loss == 'BT':
            criterion = BT_loss
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        for e in range(10):
            total_loss = 0
            ploss = 0
            eloss = 0
            for expert_traj,fail_traj in self.dataloader:
                ex_pred,ex_sampled_z = self.model(expert_traj.to(self.device))
                ex_action = expert_traj[:,self.s_dim+1:self.s_dim+self.a_dim+1].to(self.device)

                f_pred,f_sampled_z = self.model(fail_traj.to(self.device))
                f_action = fail_traj[:,self.s_dim+1:self.s_dim+self.a_dim+1].to(self.device)
                
                encoder_loss = criterion(ex_sampled_z,f_sampled_z) # encoder loss
                policy_loss = F.mse_loss(ex_pred,ex_action) + F.mse_loss(f_pred,f_action) # Policy Loss
                
                loss = encoder_loss + policy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
                eloss += encoder_loss
                ploss += policy_loss
            total_loss /= len(self.dataloader)
            eloss /= len(self.dataloader)
            ploss /= len(self.dataloader)
            strtemp = ("Epoch: %d loss: %.3f encoder loss: %.3f policy loss: %.3f")%(e,total_loss,eloss,ploss)
            print_n_txt(_chars=strtemp, _f=f)
            self.loss.append(total_loss)
            self.eloss.append(eloss)
            self.ploss.append(ploss)
        torch.save(self.model.state_dict(),self.path+'model_final.pt')

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