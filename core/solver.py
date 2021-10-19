import torch
from core.dataloader import torcs_dataset
from core.model import Encoder,Policy

class SOLVER():
    def __init__(self):
        self.device = 'cuda'
        self.load_dataset()
        self.load_model()

    def load_dataset(self):
        dataset = torcs_dataset(num_traj=3)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        self.traj_dim = 3*31
        self.a_dim = 2
        self.s_dim = 29

    def load_model(self):
        self.encoder = Encoder().to(self.device)
        self.policy = Policy().to(self.device)

    def train(self):
        for e in range(10):
            for expert_traj,fail_traj in self.dataloader:
                ex_sampled_z = self.encoder(expert_traj.to(self.device))
                ex_state = expert_traj[:,:self.s_dim].to(self.device)
                ex_action = expert_traj[:,self.s_dim+1:2*self.s_dim].to(self.device)
                inputs = torch.cat((ex_state,ex_sampled_z),dim=1)
                ex_pred = self.policy(inputs)

                f_sampled_z = self.encoder(fail_traj.to(self.device))
                f_state = fail_traj[:,:self.s_dim].to(self.device)
                f_action = fail_traj[:,self.s_dim+1:2*self.s_dim].to(self.device)
                inputs = torch.cat((f_state,f_sampled_z),dim=1)
                f_pred = self.policy(inputs)