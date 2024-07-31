# Denis
# coding:UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mine(nn.Module):
    def __init__(self, in_dim=2, hidden_size=10):
        super(Mine, self).__init__()

        self.dense1 = linear(in_dim, hidden_size)
        self.dense2 = linear(hidden_size, hidden_size)
        self.dense3 = linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        output = self.dense3(x)

        return output


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = torch.nn.Parameter(torch.normal(0.0, 0.02, size=lin.weight.shape))
    # xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.log2(torch.exp(torch.tensor(1))) * (torch.mean(t) - torch.log(torch.mean(et)))
    return mi_lb


def sample_batch(rec, noise):
    rec = torch.reshape(rec, shape=(-1, 1))
    noise = torch.reshape(noise, shape=(-1, 1))
    rec_sample1, rec_sample2 = torch.split(rec, int(rec.shape[0] / 2), dim=0)
    noise_sample1, noise_sample2 = torch.split(noise, int(noise.shape[0] / 2), dim=0)
    joint = torch.cat((rec_sample1, noise_sample1), 1)
    marg = torch.cat((rec_sample1, noise_sample2), 1)
    return joint, marg
