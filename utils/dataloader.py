import pickle
import json
import numpy as np
from numpy import random
import torch
import torch.utils.data as data

expert_dataset_name = ['expert_s.pkl','exert_a.pkl']
fail_dataset_name = ['fail_s.pkl','fail_a.pkl']
class torcs_dataset(data.Dataset):
    def __init__(self,root='./dataset/',train=True,split=0.1,num_traj=100,state_samples=20):
        self.root = root
        self.num_traj = num_traj
        self.s_dim = 29
        self.a_dim = 2
        with open(root + 'expert_s.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        expert_s = np.asarray(data) # [24588 x 29]
        indexes_e = self.define_trajectory(expert_s) # [9]
        with open(root + 'expert_a.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        expert_a = np.asarray(data) # [24588 x 2]

        with open(root + 'fail_s.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        fail_s = np.asarray(data) # [24335 x 29]
        indexes_f = self.define_trajectory(fail_s)+expert_a.shape[0] # [54]
        indexes_f = indexes_f[0::4]
        with open(root + 'fail_a.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        fail_a = np.asarray(data) # [24335 x 2]
        self.traj_indexes = np.concatenate((indexes_e,indexes_f))
        trajectories_e = np.concatenate((expert_s,expert_a),axis=-1)
        trajectories_f = np.concatenate((fail_s,fail_a),axis=-1) 
        self.trajectories = np.concatenate((trajectories_e,trajectories_f),axis=0)
        possible_index=[]
        traj_type = []
        self.total_type = self.traj_indexes.shape[0]
        for i,ind in enumerate(self.traj_indexes):
            try:
                possible_index.extend([a for a in range(ind+num_traj,self.traj_indexes[i+1]-num_traj)])
                traj_type.extend([i for _ in range(ind+num_traj,self.traj_indexes[i+1]-num_traj)])
            except:
                possible_index.extend([a for a in range(ind+num_traj,self.trajectories.shape[0]-num_traj)])
                traj_type.extend([i for _ in range(ind+num_traj,self.trajectories.shape[0]-num_traj)])
        self.possible_index = torch.LongTensor(possible_index)
        self.traj_types = torch.LongTensor(traj_type)
        self.trajectories = torch.FloatTensor(self.trajectories)
        self.traj_ = torch.unique(self.traj_types)
    
    def __getitem__(self, index):
        '''
        TODO
        start with right index! start with state
        '''
        traj_type = self.traj_[index]
        traj_index = torch.where(self.traj_types==traj_type)[0]
        perm = torch.randperm(traj_index.size(0))
        idx = perm[:2]
        samples = traj_index[idx]
        traj_1 = self.trajectories[samples[0]:samples[0]+self.num_traj,:]
        traj_2 = self.trajectories[samples[1]:samples[1]+self.num_traj,:]
        # Sample state, action  
        random_index_1 = np.random.choice(np.arange(0,self.num_traj),1)
        state_1 = traj_1[random_index_1,:self.s_dim].squeeze(0)
        action_1 = traj_1[random_index_1,self.s_dim:].squeeze(0)
        
        state_2 = traj_2[random_index_1,:self.s_dim].squeeze(0)
        action_2 = traj_2[random_index_1,self.s_dim:].squeeze(0)
        return traj_1,traj_2,state_1,action_1,state_2,action_2,traj_type
    
    def __len__(self):
        return self.traj_.size(0)
    
    def define_trajectory(self,states):
        init_state = states[0,:]
        a = np.where((states == init_state).all(axis=1))[0]
        return a

MAX_LEN = 100

class synthetic_example(data.Dataset):
    def __init__(self,path = './dataset/sdata_6.json',num_traj=50,state_samples=40):
        self.num_traj = num_traj
        self.path = path
        self.state_samples = state_samples
        with open(self.path,'r') as jf:
            data = json.load(jf)
        self.traj_type=[]
        self.traj = torch.zeros((len(data),MAX_LEN*4))
        self.traj_len=[]
        # Load Data
        for key in data:
            self.traj_type.append([data[key]['vel'],data[key]['noise'],data[key]['type']])
            traj_len = len(data[key]['states'][0])
            self.traj_len.append(traj_len)
            traj = torch.zeros((4*traj_len))
            traj[0::4] = torch.FloatTensor(data[key]['states'][0])
            traj[1::4] = torch.FloatTensor(data[key]['states'][1])
            traj[2::4] = torch.cat((torch.FloatTensor(data[key]['actions'][0]),torch.zeros(1)),dim=-1)*10
            traj[3::4] = torch.cat((torch.FloatTensor(data[key]['actions'][1]),torch.zeros(1)),dim=-1)*10
            traj = traj.repeat(int((MAX_LEN)/(traj_len)))
            self.traj[int(key),:] = traj
        self.traj_type = torch.FloatTensor(self.traj_type)
    
    def __getitem__(self, index):
        traj = self.traj[index,:]
        traj_type = self.traj_type[index]
        sampled = torch.randperm(MAX_LEN-self.num_traj)
        # sampled = random.choice([0,25,50],2)
        traj_1 = traj[4*sampled[0]:4*sampled[0]+self.num_traj*4]
        sampled_sa = torch.randperm(self.num_traj)[:self.state_samples]
        states= torch.cat((traj_1[0::4].unsqueeze(-1),traj_1[1::4].unsqueeze(-1)),dim=-1)
        actions= torch.cat((traj_1[2::4].unsqueeze(-1),traj_1[3::4].unsqueeze(-1)),dim=-1)
        state_1 = states[sampled_sa,:]
        action_1 = actions[sampled_sa,:]

        traj_2 = traj[4*sampled[1]:4*sampled[1]+self.num_traj*4]
        states= torch.cat((traj_2[0::4].unsqueeze(-1),traj_2[1::4].unsqueeze(-1)),dim=-1)
        actions= torch.cat((traj_2[2::4].unsqueeze(-1),traj_2[3::4].unsqueeze(-1)),dim=-1)
        state_2 = states[sampled_sa,:]
        action_2 = actions[sampled_sa,:]
        
        return traj_1,traj_2,state_1,action_1,state_2,action_2,traj_type
    
    def __len__(self):
        return self.traj.size(0)

if __name__ == '__main__':
    a = synthetic_example()
    b = a.__getitem__(0)
    print(b)