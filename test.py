import numpy as np
import torch
from torch import nn
from network import PPONetwork
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from gym import spaces


action_space = spaces.Box(low=-1, high=1, shape=(3,))
state_space = spaces.Box(low=-1, high=1, shape=(6,))

# print(action_space.shape)
network = PPONetwork(action_space=action_space, in_size=state_space.shape[0])
#print(network(torch.ones((10,6)) ))
s1 = torch.tensor([1,2,3,4,5,6],dtype=torch.float32)
s2 = torch.tensor([0,1,3,4,5,6],dtype=torch.float32)

new_prob = network( torch.stack([s1,s2]) )
# print( torch.eye(3) * new_prob[:,action_space.shape[0]:] )
S = torch.zeros((2,9))
S[:,np.array([0,4,8])] = new_prob[:,action_space.shape[0]:]
S = S.reshape(2,3,3)

# print(new_prob[:,action_space.shape[0]:])
# print(S)

new_dist_buffer = MultivariateNormal(new_prob[:,:action_space.shape[0]],  S )
#new_dist_buffer = Normal(loc=new_prob[:,:action_space.shape[0]], scale=new_prob[:,action_space.shape[0]:], validate_args=True)
action = new_dist_buffer.sample()
# print(n.log_prob( actions ))

# print(new_dist_buffer.log_prob(action))
# print(n.log_prob(action))


# print(new_dist_buffer.sample() )
# action = new_dist_buffer.sample()
log_buffer = [new_dist_buffer.log_prob(action[0]).reshape(-1), new_dist_buffer.log_prob(action[1]).reshape(-1)]
# print(torch.stack(log_buffer).shape)
dist_buffer = torch.exp(torch.stack(log_buffer).detach())
# print(di)

rho = new_dist_buffer.log_prob(torch.stack([action[0].reshape(-1),action[1].reshape(-1)]))
e = 0.5
m = torch.tensor(-e,dtype=torch.float32)
M = torch.tensor(+e,dtype=torch.float32)
print(rho)
# print(torch.sum(rho,axis=1))
# print(torch.min(torch.max(m, rho),M))
a = rho*torch.tensor([1,2],dtype=torch.float32)
print(torch.min(a, rho))
