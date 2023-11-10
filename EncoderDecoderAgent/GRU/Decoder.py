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
    def __init__(self, hidden_size, action_length=3):
        """
        :param hidden_size: size of the hidden output from attention layer
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        self.policy_network = nn.Sequential(
            NoisyLinear(hidden_size, 128),
            nn.BatchNorm1d(128),
            NoisyLinear(128, 256),
            nn.BatchNorm1d(256),
            NoisyLinear(256, action_length))

    def forward(self, x):

        x = x.squeeze().unsqueeze(0) if len(x.squeeze().shape) < 2 else x.squeeze()
        output = self.policy_network(x).squeeze()
        return output if len(output.shape) > 1 else output.unsqueeze(0)
