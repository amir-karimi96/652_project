import numpy as np
import torch
from torch import nn
from network import PPONetwork
from torch.distributions.normal import Normal

from gym import spaces


action_space = spaces.Box(low=-1, high=1, shape=(3,))
state_space = spaces.Box(low=-1, high=1, shape=(6,))

print(action_space.shape)
network = PPONetwork(action_space=action_space, in_size=state_space.shape[0])
#print(network(torch.ones((10,6)) ))
a = torch.rand((10,6))
new_prob = network( a )
print(new_prob)
new_dist_buffer = Normal(loc=new_prob[:,:action_space.shape[0]], scale=new_prob[:,action_space.shape[0]:], validate_args=True)

#print(new_dist_buffer.log_prob(new_prob[:,:-1]))
