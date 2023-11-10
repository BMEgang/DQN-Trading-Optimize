# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DQN(nn.Module):
#
#     def __init__(self, state_length, action_length):
#         super(DQN, self).__init__()
#         self.policy_network = nn.Sequential(
#             nn.Linear(state_length, 128),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, action_length))
#
#         # self.layer1 = nn.Linear(state_length, 128)
#         # self.bn1 = nn.BatchNorm1d(128)
#         # self.layer2 = nn.Linear(128, 256)
#         # self.bn2 = nn.BatchNorm1d(256)
#         # self.out = nn.Linear(256, action_length)
#
#     def forward(self, x):
#         # if x.shape[0] > 1:
#         #     x = F.relu(self.bn1(self.layer1(x)))
#         #     x = F.relu(self.bn2(self.layer2(x)))
#         # else:
#         #     x = F.relu(self.layer1(x))
#         #     x = F.relu(self.layer2(x))
#         # return self.out(x)
#         return self.policy_network(x)
import torch.nn as nn
import torch.nn.functional as F
import torch

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.noise_std = std_init / (in_features ** 0.5)
        self.register_buffer('weight_noise', torch.empty(out_features, in_features))
        self.register_buffer('bias_noise', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def reset_noise(self):
        noise = self._scale_noise(self.in_features)
        noise = noise.expand(self.out_features, noise.size(0))
        self.weight_noise.copy_(noise)
        self.bias_noise.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        noise = torch.randn(size)
        return noise.sign().mul_(noise.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight + self.weight_noise, self.bias + self.bias_noise)
        return F.linear(x, self.weight, self.bias)


class Q_NetWork(nn.Module):

    def __init__(self, state_length, action_length, hidden=[128, 256]):
        super(Q_NetWork, self).__init__()

        self.feature = nn.Sequential(
            NoisyLinear(state_length, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            NoisyLinear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1])
        )

        self.advantage = nn.Linear(hidden[1], action_length)
        self.value = nn.Linear(hidden[1], 1)

    def forward(self, x):
        x = self.feature(x)

        advantage = self.advantage(x)
        value = self.value(x)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
