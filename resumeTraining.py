#!/usr/bin/env python
# coding: utf-8

# In[67]:


import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
from tf_agents.environments import suite_gym
import matplotlib.pyplot as plt
from tf_agents.environments.wrappers import ActionRepeat
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.networks.q_network import QNetwork
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.environments import suite_gym


# In[68]:


checkpoint_dir= "agent/"


# In[69]:


os.listdir(checkpoint_dir)


# In[70]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[71]:


model = tf.keras.Model(...)
checkpoint=tf.train.Checkpoint(optimizer=optimizer,model=model)


# In[82]:


agent = checkpoint.restore(latest)


# In[83]:


max_episode_steps = 27000 
environment_name = "SpaceInvadersNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])


# In[84]:


from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)


# In[85]:


from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.)


# In[86]:


from tf_agents.metrics import tf_metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
    ]


# In[81]:


from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)


# In[76]:


from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)
checkpoint = tf.train.Checkpoint(agent)


# In[88]:


def train_agent(n_iterations):
    time_step = None
    
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            save_path = checkpoint.save('agent/lastModelCheckpoint')
    


# In[89]:


train_agent(n_iterations=1000)


# In[ ]:





# In[ ]:




