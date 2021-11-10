import torch
import matplotlib.pyplot as plt

device = 'cuda'

net.eval()
traj_x = []
traj_y = []
x = 1.
y = 2.
for i in range(50):
    traj_x.append(x)
    traj_y.append(y)
    with torch.no_grad():
        input = torch.FloatTensor([x,y]).unsqueeze(0)
        prediction = net(input.to(device))
        dx = prediction[0][0].cpu().item()
        dy = prediction[0][1].cpu().item()
        x += dx
        y += dy
plt.figure()
plt.ylim(0,4)
plt.xlim(-2,2)
plt.scatter(traj_x,traj_y)
plt.show()