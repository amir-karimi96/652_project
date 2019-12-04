#!/usr/bin/env python

import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter


from ppo_agent import PPO

import time
import sys
sys.path.append('/home/amir/RL_LAB/SenseAct/')

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.devices.ur import ur_utils
from senseact.utils import NormalizedEnv

import subprocess
from network import PPONetwork

import gym
from torch import nn

from matplotlib import pyplot as plt
import numpy as np
def main(cycle_time, idn, baud, port_str, batch_size, mini_batch_div, epoch_count, gamma, l, max_action, outdir,
         ep_time,index):
    """
    :param cycle_time: sense-act cycle time
    :param idn: dynamixel motor id
    :param baud: dynamixel baud
    :param batch_size: How many sample to record for each learning update
    :param mini_batch_size: How many samples to sample from each batch
    :param epoch_count: Number of epochs to train each batch on. Is this the number of mini-batches?
    :param gamma: Usual discount value
    :param l: lambda value for lambda returns.


    In the original paper PPO runs N agents each collecting T samples.
    I need to think about how environment resets are going to work. To calculate things correctly we'd technically
    need to run out the episodes to termination. How should we handle termination? We might want to have a max number
    of steps. In our setting we're going to be following a sine wave - I don't see any need to terminate then. So we
    don't need to run this in an episodic fashion, we'll do a continuing task. We'll collect a total of T samples and
    then do an update. I think I will implement the environment as a gym environment just to permit some
    interoperability. If there was an env that had a terminal then we would just track that terminal and reset the env
    and carry on collecting. Hmmm, actually I'm not sure how to think about this as a gym env. So SenseAct uses this
    RTRLBaseEnv, but I'm not sure I want to do that.

    So the changes listed from REINFORCE:
    1. Drop γ^t from the update, but not from G_t
    2. Batch Updates
    3. Multiple Epochs over the same batch
    4. Mini-batch updates
    5. Surrogate objective: - π_θ/π_θ_{old} * G_t
    6. Add Baseline
    7. Use λ-return: can you the real lambda returns or use generalized advantage estimation like they do in the paper.
    8. Normalize the advantage estimates: H = G^λ - v
    9. Proximity constraint:
        ρ = π_θ/π_θ_{old}
        objective:
        -min[ρΗ, clip(ρ, 1-ε, 1+ε)H]

    Also, there is the value function loss and there is an entropy bonus given.

    """
    #set low latency for usb-serial communications
#    bashCommand = "setserial /dev/ttyUSB0 low_latency"
#    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate()
    #bashCommand = "cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer"
    #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
#    print(output)

    tag = f"{time.time()}"
    summaries_dir = f"./summaries/{tag}"
    returns_dir = "./returns"
    networks_dir = "./networks"
    if outdir:
        summaries_dir = os.path.join(outdir, f"summaries/{tag}")
        returns_dir = os.path.join(outdir, "returns")
        networks_dir = os.path.join(outdir, "networks")

    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(returns_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=summaries_dir)

    #env = ReacherEnv(cycle_time, ep_time, dxl.get_driver(False), idn, port_str, baud, max_action,'tourq')
    # env = ReacherEnv(setup='UR5_default',
    #                 host='129.128.159.210',
    #                 dof=2,
    #                 control_type='position',
    #                 derivative_type='none',
    #                 target_type='position',
    #                 reset_type='random',
    #                 reward_type='linear',
    #                 deriv_action_max=10,
    #                 first_deriv_max=10,
    #                 vel_penalty=0,
    #                 obs_history=1,
    #                 actuation_sync_period=1,
    #                 episode_length_time=4.0,
    #                 episode_length_step=None,
    #                 rllab_box = False,
    #                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
    #                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
    #                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
    #                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
    #                 movej_t=2,
    #                 accel_max=None,
    #                 speed_max=None,
    #                 dt=0.008,
    #                 delay=0.0)

    #DO WE NEED RANDOM STATE Variable??
    rand_state = np.random.RandomState(1).get_state()
    host_ip = '169.254.39.68'

    env = ReacherEnv(
        setup="UR5_default",
        host = host_ip,
        dof=2,
        control_type="velocity",
        target_type="position",
        reset_type="zero",
        reward_type="precision",
        derivative_type="none",
        deriv_action_max=5,
        first_deriv_max=2,
        accel_max=1.4,
        speed_max=0.3,
        speedj_a=1.4,
        episode_length_time=4.0,
        episode_length_step=None,
        actuation_sync_period=1,
        dt=0.04,
        run_mode="singlethread",
        rllab_box=False,
        movej_t=2.0,
        delay=0.0,
        random_state=rand_state
    )
    #print('done')
    env = NormalizedEnv(env)
    env.start()
    #print("starting")
#    obs = env.reset()
#    print('resetted', obs)
#    env.step(action=np.array([0,0])
#    print('a')
#    time.sleep(10)
    #env = gym.make('MountainCarContinuous-v0')
    obs_len = env.observation_space.shape[0]
    print(env.action_space.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_network = PPONetwork(action_space=env.action_space, in_size=env.observation_space.shape[0])  # TODO: create your network
    ppo_network.to(device)

    # instanciate  value_network
    value_network = nn.Sequential(nn.Linear(obs_len, 50),
                                    nn.Sigmoid(),
                                    nn.Linear(50,1))
    value_network.to(device)

    # instanciate the agent
    agent = PPO(device=device, network=ppo_network, state_size=obs_len, batch_size=batch_size,
                mini_batch_div=mini_batch_div, epoch_count=epoch_count, gamma=gamma, l=l, eps=0.2, summary_writer = summary_writer, value_network= value_network)
    # TODO: implement your main loop here. You will want to collect batches of transitions
    #

    # total number of timesteps
    t = 0
    #total_steps = 1000
    #timestep_per_episode = 200
    n_batch = 36
    undiscounted_return = np.zeros((n_batch,batch_size))
    # do learning for a number of total timesteps
    for b in range(n_batch):#total_steps // timestep_per_episode):
        print(b)

        # gather batch of episodes
        for ep in range( batch_size ):
            # reset the env before each episode
            observation = env.reset()
            reward = 0
            n = 0

            # gather one episone
            while(True):
                #if b > 90 :
                #    env.render()
                action = agent.step(state = observation, r = reward, t=n)
                action = action * max_action
                observation, reward, done, info = env.step((action,)) # take the action
                #print(observation)
                undiscounted_return[b,ep] = undiscounted_return[b,ep]+reward
                #print(action,reward)
                t = t + 1
                n = n+1
                if done:
                    break
            # end of one episode

        # learning using batch of data
        summary_writer.add_scalar('return', np.mean(undiscounted_return[b,:]), 2048*b)
        env.stop()
        agent.learn(t = t)
        agent.reset_buffers()
        t = 0

    env.close()
    # ploting results
    undiscounted_return_avg = np.mean(undiscounted_return, axis=1)
    np.save('ep_returns_{}'.format(index),undiscounted_return_avg)
    plt.plot(undiscounted_return_avg)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.04, help="sense-act cycle time")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--batch_size", type=int, default=41,
                        help="How many episodes to record for each learning update")
    parser.add_argument("--mini_batch_div", type=int, default=32, help="Number of division to divide batch into")
    parser.add_argument("--epoch_count", type=int, default=10,
                        help="Number of times to train over the entire batch per update.")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount")
    parser.add_argument("--l", type=float, default=0.99, help="lambda for lambda return")
    parser.add_argument("--max_action", type=float, default=0.3,
                        help="The maximum value you will output to the motor. "
                             "This should be dependent on the control mode which you select.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")
    parser.add_argument("--index","-i", type=int, default=0, help="run index")
    args = parser.parse_args()
    main(**args.__dict__)
