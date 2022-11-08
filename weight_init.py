import torch
import torch.nn as nn
from Nets import Net_NonLinReg


def init_uniform(module):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=-0.01, b=0.01)
        nn.init.zeros_(module.bias)

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.1)
        nn.init.zeros_(module.bias)

def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, val=0)
        nn.init.zeros_(module.bias)

# xavier behaves well in tanh, but is not applicable to ReLU
def init_xavier_uniform(module):
    # ~U(-a, a), a = gain * sqrt(6 / (fan_in + fan_out))
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.zeros_(module.bias)

def init_xavier_normal(module):
    # ~N(0, std), std = gain * sqrt(2 / (fan_in + fan_out))
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=1)
        nn.init.zeros_(module.bias)

# He intialization
# fan_in keep variance of weights changeless in forward propagation
# a: negative slope of activation function in next layer (default 0, i.e. relu)
def init_kaiming_uniform(module):
    # ~U(-a, a), a = sqrt(6 / ((1 + a^2) * fan_in))
    if type(module) == nn.Linear:
        nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)

def init_kaiming_normal(module):
    # ~N(0, std), std = sqrt(2 / ((1 + a^2) * fan_in))
    if type(module) == nn.Linear:
        nn.init.kaiming_uniform_(module.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(module.bias)

if __name__ == '__main__':

    net1 = Net_NonLinReg(input_dim=1, n_hidden=10, out_dim=1, dropout_rate=0)
    net1.apply(init_uniform)
    for param in net1.parameters():
        print (param)

    print ('%%%%%%')
    net2 = Net_NonLinReg(input_dim=1, n_hidden=10, out_dim=1, dropout_rate=0)
    net2.apply(init_uniform)
    for param in net2.parameters():
        print (param)

