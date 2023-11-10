import torch.nn as nn
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


class Decoder(nn.Module):
    def __init__(self, num_classes, action_length=3):
        super(Decoder, self).__init__()

        # Feature extraction with NoisyNet
        self.feature = nn.Sequential(
            NoisyLinear(num_classes, 128),
            nn.BatchNorm1d(128),
            NoisyLinear(128, 256),
            nn.BatchNorm1d(256)
        )

        # Dueling Network structure
        self.advantage = NoisyLinear(256, action_length)
        self.value = NoisyLinear(256, 1)

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        x = self.feature(x)

        advantage = self.advantage(x)
        value = self.value(x)

        # Combine them to get Q-values
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# class Decoder(nn.Module):
#     def __init__(self, input_size, action_length=3):
#         """
#
#         :param state_length: we give OHLC as input to the network
#         :param action_length: Buy, Sell, Idle
#         """
#         super(Decoder, self).__init__()
#
#         self.policy_network = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, action_length))
#
#     def forward(self, x):
#         if len(x.shape) < 2:
#             x = x.unsqueeze(0)
#         return self.policy_network(x)
