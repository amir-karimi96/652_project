"""
Place your PPO agent code in here.
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

class PPO:
    def __init__(self,
                 device,  # cpu or cuda
                 network,  # your network
                 state_size,  # size of your state vector
                 batch_size,  # size of batch
                 mini_batch_div,
                 epoch_count,
                 gamma=0.99,  # discounting
                 l=0.95,  # lambda used in lambda-return
                 eps=0.2,  # epsilon value used in PPO clipping
                 summary_writer: SummaryWriter = None,
                 value_network=None ):  #value network):

        self.device = device

        self.batch_size = batch_size
        self.mini_batch_div = mini_batch_div
        self.mini_batch_size = 0# is calculated in self.learn
        self.epoch_count = epoch_count
        self.gamma = gamma
        self.l = l
        self.eps = eps
        self.summary_writer = summary_writer

        self.state_size = state_size
        self.network = network
        self.value_network = value_network
        self.optimizer = torch.optim.Adam(self.network.parameters())#, lr=1e-5)
        self.optim_value = torch.optim.Adam(params = self.value_network.parameters())

        # new variables
        self.update_counter = 0
        self.timestep_buffer = []
        self.s_buffer = []
        self.r_buffer = []
        self.v_buffer = []
        self.a_buffer = []
        self.G_buffer = []
        self.H_buffer = []
        self.dist_old_buffer = []
        self.dist_log_old_buffer = []
        return None

    @staticmethod
    def _segment_network(output):
        actions = output[:len(output)//2]
        stds = output[len(output)//2:]

        return actions, stds

    @staticmethod
    def _segment_network_2(output):
        mus = output[:,:output.shape[1]//2]
        stds = output[:,output.shape[1]//2:]

    @staticmethod
    def _array_to_sigma_matrix(array):
        """
        array of std shape(#states, action_count)
        return [sigma1_3x3, ... sigman_3x3]
        """
        S = torch.zeros((array.shape[0],9))
        S[:,np.array([0,4,8])] = array
        S = S.reshape(array.shape[0],3,3)
        return S

    def step(self, state, r, t):
        """
        You will need some step function which returns the action.
        This is where I saved my transition data in my own code.
        :param state:
        :param r:
        :param t: current timestep
        :return: action
        """
        self.timestep_buffer.append(t)

        output = self.network(torch.tensor(state, device=self.device, dtype=torch.float32))

        #A to have always two dimensional output even with one input
        output_2 = self.network(torch.tensor(state, device=self.device, dtype=torch.float32).reshape(1,self.state_size))

        #output in format
        # joint 1 action, joint 2 action, ... joint 1 std, joint 2 std, ....
        mus, stds = self._segment_network(output)

        #A second version for array or one input
        mus, std = self._segment_network_2(output_2)

        # This is the policy network. Why is it value???
        #A not having summaries for now
        #self.summary_writer.add_scalar('policy/mus/', mus.detach().numpy(), len(self.timestep_buffer))


        # add a very small amount to make sure std is not exactly zero
        STDS = self._array_to_sigma_matrix(std)
        self.dist_old_buffer.append( Normal(loc = mus, scale = STDS, validate_args=True))

        # sampling action shape (1, action_count )  --> (action_count, )
        actions = self.dist_old_buffer[-1].sample().reshape(-1)

        # saving log probability
        self.dist_log_old_buffer.append( torch.sum(self.dist_old_buffer[-1].log_prob(actions).reshape(-1)))

        # saving transition data
        # reward for first time step is zero
        self.r_buffer.append(r)
        self.s_buffer.append(torch.tensor(state, device=self.device, dtype=torch.float32)) #note that last state is not saved
        self.a_buffer.append(torch.tensor(actions.numpy(), device=self.device, dtype=torch.float32))

        return actions.numpy()

    @staticmethod
    def compute_return(r_buffer, v_buffer, t_buffer, l=0, gamma=0):
        """

        Compute the return. Unit test this function

        :param r_buffer: rewards like [0, r1 r2 ...rT1,0 r1 r2 ...rT2 ...rTn]
        :param v_buffer: values       [v(s0) v(s1).. v(sT) ... v(s0) v(s1).. v(sTn)]
        :param t_buffer: time steps [0 1 2 3 ... T1 0 1 2 3 ...Tn]
        :param l: lambda value
        :param gamma: gamma value
        :return:  the return
        """
        G = np.zeros(t_buffer.shape)
        T = t_buffer.copy()
        T[:-1] = T[:-1]-T[1:]
        # finding terminals
        terminal_ind = np.where(T>=0)[0]

        # for each episode in buffers
        for i in range(len(terminal_ind)):
            # recursivly computing return
            for t in np.arange(terminal_ind[i]-1,terminal_ind[i]-t_buffer[terminal_ind[i]]-1,-1):
                #PPO7 compute lambda return
                G[t] = r_buffer[t+1] + gamma*( (1-l)*v_buffer[t+1] + l*G[t+1])

        return G

    def compute_advantage(self, g, v):
        """
        Compute the advantage
        :return: the advantage
        """
        return g-v

    def compute_rho(self, actions, old_pi, new_pi):
        """
        Compute the ratio between old and new pi
        :param actions: torch stack of actions
        :param old_pi: numpy array of pdf of actions shape (#states, )
        :param new_pi: torch stack of normal objects shape (#states, )
        :return: rho_buffer shape (#states, )
        """
        return torch.exp( new_pi.log_prob(actions) ) / old_pi

    def clip(self,rho_buffer,e):
        """
        rho_buffer : shape(#states, action_count)
        return : clipped rho_buffer of shape(#states,)
        """
        m = torch.tensor(1-e,dtype=torch.float32)
        M = torch.tensor(1+e,dtype=torch.float32)
        return torch.min(torch.max(m, rho_buffer),M)

    def learn(self, t):
        """
        Here's where you should do your learning and logging.
        :param t: The total number of transitions observed. used for learning
        :PPO_step:
        :return:
        """
        # now there is B new episodes observed for learning
        # state action and reward buffers are ready
        # we are sure that t = B * T
        T = t/ self.batch_size
        T = int(T)
        self.timestep_buffer = np.array(self.timestep_buffer)
        self.v_buffer = self.value_network(torch.stack(self.s_buffer)).reshape(t)



        # used for value update
        self.G_buffer = self.compute_return( r_buffer= self.r_buffer,
                                        v_buffer = self.v_buffer.detach().numpy(),
                                        t_buffer=self.timestep_buffer,
                                        l=1,
                                        gamma=self.gamma)
        # lambda return
        self.G_l_buffer = self.compute_return( r_buffer= self.r_buffer,
                                        v_buffer = self.v_buffer.detach().numpy(),
                                        t_buffer=self.timestep_buffer,
                                        l=self.l,
                                        gamma=self.gamma)
        #PPO2 compute advantage ( adding baseline)
        self.H_buffer = torch.tensor(self.compute_advantage(g=self.G_l_buffer, v=self.v_buffer.detach().numpy()),dtype=torch.float32)

        #PPO8 normalize advantage
        self.H_buffer = (self.H_buffer - torch.mean(self.H_buffer))/torch.std(self.H_buffer)


        # regenerate old_pi from old_log_buffer
        # we want just the numbers not grads so calling detach
        self.dist_old_buffer = torch.exp(torch.stack(self.dist_log_old_buffer).detach())

        #PPO5 Multiple Epochs
        for e in range(self.epoch_count):
            # shuffle needed buffers
            shuffled_ind = np.random.permutation(t)
            shuffled_s_buffer = torch.stack(self.s_buffer)[shuffled_ind]
            shuffled_a_buffer = torch.stack(self.a_buffer)[shuffled_ind]
            shuffled_H_buffer = self.H_buffer[shuffled_ind]

            shuffled_dist_old_buffer = self.dist_old_buffer[shuffled_ind]
            self.mini_batch_size = t // self.mini_batch_div

            #PPO4 Mini_batch updates
            for m in range(self.mini_batch_div):
                division_ind = np.arange(m*self.mini_batch_size , (m+1)*self.mini_batch_size)


                output = self.network(shuffled_s_buffer[division_ind])
                mus, stds = self._segment_network_2(output)
                STDS = self._array_to_sigma_matrix(stds)
                new_dist_buffer = Normal(loc = mus, scale = stds, validate_args=True)

                #PPO6 compute rho coefficients
                rho_buffer = self.compute_rho(actions=shuffled_a_buffer[division_ind],
                                        old_pi=shuffled_dist_old_buffer[division_ind],
                                        new_pi=new_dist_buffer)

                #PPO1 droping gamma from loss
                #PPO9 apply changing penalty
                self.loss_buffer = - torch.min(rho_buffer * shuffled_H_buffer[division_ind].detach(),
                                                self.clip(rho_buffer, self.eps) * shuffled_H_buffer[division_ind].detach())
                #PPO3 batch update
                self.loss_policy = torch.sum(self.loss_buffer)/self.mini_batch_size

                # compute delta_buffer = (g-v) for value update
                self.v_buffer = self.value_network(torch.stack(self.s_buffer)).reshape(t)[shuffled_ind]

                delta_buffer = self.compute_advantage(g = torch.tensor(self.G_buffer[shuffled_ind], dtype=torch.float32,device=self.device),
                                                        v=self.v_buffer)
                self.loss_value = torch.sum(delta_buffer[division_ind]**2)/self.mini_batch_size
                self.optimize_step()


        return 0


    def optimize_step(self):
        self.update_counter += 1
        self.loss = self.loss_policy + self.loss_value
        self.optim_value.zero_grad()
        self.optimizer.zero_grad()
        self.loss.backward()

        #self.summary_writer.arr('actions', self.a_buffer , ep)
        self.summary_writer.add_scalar('Loss/policy',self.loss_policy.detach().numpy(), self.update_counter)
        self.summary_writer.add_scalar('Loss/value',self.loss_value.detach().numpy(), self.update_counter)
        self.summary_writer.add_scalar('Loss/total', self.loss.detach().numpy(), self.update_counter)
        self.summary_writer.add_scalar('value/sigma.value', torch.mean(self.network.sigma_sqrt.weight).detach().numpy(), self.update_counter)
        #self.summary_writer.add_scalar('grad/mu.bias', torch.mean((self.network.mu[2].bias.grad)**2).numpy(), self.update_counter)
        self.summary_writer.add_scalar('grad/mu.weight', torch.mean((self.network.mu[2].weight.grad)**2).numpy(), self.update_counter)
        self.summary_writer.add_scalar('grad/value.bias', torch.mean((self.value_network[2].bias.grad)**2).numpy(), self.update_counter)
        self.summary_writer.add_scalar('grad/value.weight', torch.mean((self.value_network[2].weight.grad)**2).numpy(), self.update_counter)

        self.optimizer.step()
        self.optim_value.step()

    def reset_buffers(self):
        """
        resets all the agents' buffers to zero for next learning phase
        inputs:
        return:
        """
        self.timestep_buffer = []
        self.s_buffer = []
        self.r_buffer = []
        self.v_buffer = []
        self.a_buffer = []
        self.G_buffer = []
        self.H_buffer = []
        self.dist_old_buffer = []
        self.dist_log_old_buffer = []
