import pickle
import numpy as np
import torch
import torch.utils.data as data

expert_dataset_name = ['expert_s.pkl','exert_a.pkl']
fail_dataset_name = ['fail_s.pkl','fail_a.pkl']
class torcs_dataset(data.Dataset):
    def __init__(self,root='./dataset/',train=True,split=0.1,num_traj=3):
        self.root = root
        with open(root + 'expert_s.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        expert_s = np.asarray(data) # [24584 x 29]
        
        with open(root + 'expert_a.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        expert_a = np.asarray(data) # [24584 x 2]

        with open(root + 'fail_s.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        fail_s = np.asarray(data) # [24335 x 29]
        
        with open(root + 'fail_a.pkl', 'rb') as f:
            data = pickle.load(f)
        f.close()
        fail_a = np.asarray(data) # [24335 x 2]

        expert_traj = np.concatenate((expert_s[:-num_traj-1],expert_a[:-num_traj-1]),axis=1)
        fail_traj = np.concatenate((fail_s[:-num_traj-1],fail_a[:-num_traj-1]),axis=1)

        for j in range(1,num_traj):
            expert_traj = np.concatenate((expert_traj,expert_s[j:j-num_traj-1]),axis=1)
            expert_traj = np.concatenate((expert_traj,expert_a[j:j-num_traj-1]),axis=1)
            fail_traj = np.concatenate((fail_traj,fail_s[j:j-num_traj-1]),axis=1)
            fail_traj = np.concatenate((fail_traj,fail_a[j:j-num_traj-1]),axis=1)
        
        self.exper_traj = torch.FloatTensor(expert_traj)
        self.fail_traj = torch.FloatTensor(fail_traj)
        
        num_expert = self.exper_traj.size(0)
        num_fail = self.fail_traj.size(0)
        # Shuffle
        rand_idx = torch.randperm(num_expert)
        self.exper_traj = self.exper_traj[rand_idx]
        rand_idx = torch.randperm(num_fail)
        self.fail_traj = self.fail_traj[rand_idx]
    
        # Train Test Split
        if not train:
            self.exper_traj = self.exper_traj[:int(num_expert*split)]
            self.fail_traj = self.fail_traj[:int(num_fail*split)]
        else:
            self.exper_traj = self.exper_traj[int(num_expert*split):]
            self.fail_traj = self.fail_traj[int(num_fail*split):]
        
        self.num_expert = self.exper_traj.size(0)
        self.num_fail = self.fail_traj.size(0)
        
        self.min_value = min(self.num_fail,self.num_expert)

    def __getitem__(self, index):
        fail_index = index%self.min_value
        return self.exper_traj[index],self.fail_traj[fail_index]
    
    def __len__(self):
        return max(self.num_expert,self.num_fail)

if __name__ == '__main__':
    a = torcs_dataset()