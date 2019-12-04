import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal


# TODO: create your network here. Your network should inherit from nn.Module.
# It is recommended that your policy and value networks not share the same core network. This can be
# done easily within the same class or you can create separate classes.


class PPONetwork(nn.Module):
    def __init__(self, action_space, in_size):
        """
        Feel free to modify as you like.

        The policy should be parameterized by a normal distribution (torch.distributions.normal.Normal).
        To be clear your policy network should output the mean and stddev which are then fed into the Normal which
        can then be sampled. Care should be given to how your network outputs the stddev and how it is initialized.
        Hint: stddev should not be negative, but there are numerous ways to handle that. Large values of stddev will
        be problematic for learning.

        :param action_space: Action space of the environment. Gym action space. May have more than one action.
        :param in_size: Size of the input
        """
        super(PPONetwork, self).__init__()
        self.action_count = action_space.shape[0]
        self.in_size = in_size
        self.model = nn.Sequential(nn.Linear(self.in_size,64),
                            nn.Sigmoid(),
                            nn.Linear(64,64),   # 1 output for each action here just one
                            nn.Sigmoid(),
                            nn.Linear(64,2*self.action_count),
                            )
        #self.sigma_sqrt = nn.Linear(1, self.action_count, bias=False)
        # initializ sigma to sqrt(0.5) ~ 0.7
        self.Tanh = nn.Tanh()
        self.Softplus = nn.Softplus()
        #self.sigma_sqrt.weight = nn.Parameter(0.7*torch.ones((self.action_count,1)))

    def forward(self, inputs):
        '''
        inputs shape is (#states, obs_len)
        return [mus sigmas] with shape (#states, 2 * action_count )
        '''
        out = self.model(inputs)
        mu = self.Tanh(out[:,:self.action_count])
        sigma = self.Softplus(out[:,self.action_count:])

        #if mu.shape[0] == 1:
        #    sigma = self.sigma_sqrt(torch.ones([1]))**2
        #    return torch.cat((mu, sigma ))

        #sigma = self.sigma_sqrt(torch.ones((mu.shape[0],1)))**2
        return torch.cat((mu, sigma ),1)
