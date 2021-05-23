import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

def compute_mutual_information(x, y, device, lr=1e-4, batch_size = 100, iter_num = int(5), log_freq = int(1e+3), ma_et = 1., ma_rate = 0.01):

    bs = x.shape[0]
    feature = x.shape[1]
    result = list()

    mine_net = Mine(input_size=feature * 2).to(device)            # x(y).shape: [bs, feature]
    mine_net_optimal = optim.AdamW(mine_net.parameters(), lr)       # MINE shape: feature * 2 -> 1

    for i in range(iter_num):

        joint_index = np.random.choice(range(bs), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(bs), size=batch_size, replace=False)
        joint_batch = torch.cat((x[joint_index], y[joint_index]), 1)         # joint_batch.shape: [bs, feature * 2]
        marginal_batch = torch.cat((x[joint_index], y[marginal_index]), 1)

        t = mine_net(joint_batch)
        et = torch.exp(mine_net(marginal_batch))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))  # Evaluate the lower-bound
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et) # moving average

        mi_loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))

        mine_net_optimal.zero_grad()
        autograd.backward(mi_loss)
        mine_net_optimal.step()

        result.append(mi_lb.detach().cpu().numpy())

        if (i+1)%(log_freq)==0:
            print(result[-1])
    print(result[-1])
    return mine_net


def compute_mi(mine_net, x, y, sample_size=100):

    joint_index = np.random.choice(range(x.shape[0]), size=sample_size, replace=False)
    marginal_index = np.random.choice(range(x.shape[0]), size=sample_size, replace=False)

    joint_batch = torch.cat((x[joint_index], y[joint_index]), 1)  # joint_batch.shape: [bs, feature * 2]
    marginal_batch = torch.cat((x[joint_index], y[marginal_index]), 1)

    t = mine_net(joint_batch)
    et = torch.exp(mine_net(marginal_batch))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))  # Evaluate the lower-bound

    return mi_lb


if __name__ == "__main__":

    device = torch.device("cuda:{}".format(0))

    x = np.random.multivariate_normal(mean=[0, 0],
                                      cov=[[1, 0.8], [0.8, 1]],
                                      size=600)
    print("x.shape :", x.shape)
    y = np.random.multivariate_normal(mean=[0, 0],
                                      cov=[[1, 0.8], [0.8, 1]],
                                      size=600)
    print("y.shape :", y.shape)
    x = torch.tensor(x).to(dtype=torch.float32).to(device)
    y = torch.tensor(y).to(dtype=torch.float32).to(device)

    compute_mutual_information(x ,x, device)