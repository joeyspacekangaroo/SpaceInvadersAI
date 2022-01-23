#!/usr/bin/env python
# coding: utf-8

# In[146]:


import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os


# In[147]:


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[148]:



def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


# In[149]:


atariGameTitles=[]


# In[150]:


for z in gym.envs.registry.all():
    envTitle=str(z).partition(')')[0].partition('(')[2]
    if envTitle.count('-ram')>0:
        if atariGameTitles.count(envTitle.partition('-ram')[0])==0:
            atariGameTitles.append(envTitle.partition('-ram')[0])
print('The Atari Games Available Are')
print(atariGameTitles)


# In[151]:


for gameTitle in atariGameTitles:
    if gameTitle == 'SpaceInvaders':
        #Defender either does not work or takes too long
        myEnv=gym.make(gameTitle+'-v4')
        print('Action Space for '+gameTitle+'-v4 is '+str(myEnv.action_space)+' corresponding to:')
        print(myEnv.get_action_meanings())
        print()


# In[152]:


from tf_agents.environments import suite_gym


# In[153]:


env=suite_gym.load('SpaceInvaders-v4')


# In[154]:


env.reset()


# In[155]:


env.step(1) # fire


# In[156]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img = env.render(mode="rgb_array")
plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis("off")


# In[157]:


env.observation_spec()


# In[158]:


env.action_spec()


# In[159]:


env.time_step_spec()


# In[160]:


from tf_agents.environments.wrappers import ActionRepeat
from gym.wrappers import TimeLimit


# In[161]:


import tf_agents.environments.wrappers
for name in dir(tf_agents.environments.wrappers):
    obj = getattr(tf_agents.environments.wrappers, name)
    if hasattr(obj, "__base__") and issubclass(obj, tf_agents.environments.wrappers.PyEnvironmentBaseWrapper):
        print("{:27s} {}".format(name, obj.__doc__.split("\n")[0]))


# In[162]:


myRepEnv=ActionRepeat(env,times=4)


# In[163]:


myLimRepEnv=suite_gym.load('SpaceInvaders-v4',
    gym_env_wrappers=[lambda env: TimeLimit(env,max_episode_steps=10000)],
    env_wrappers=[lambda env: ActionRepeat(env,times=4)])


# In[164]:


from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4


# In[165]:


max_episode_steps = 27000 
environment_name = "SpaceInvadersNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])


# In[166]:


env


# In[167]:


env.seed(42)
env.reset()
for _ in range(4):
    time_step = env.step(np.array(2)) # RIGHT

def plot_observation(obs):
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")
plt.figure(figsize=(6, 6))
plot_observation(time_step.observation)
plt.show()


# In[168]:


from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)


# In[169]:


from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.)


# In[170]:


conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]


q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)


# In[171]:


from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95,
                momentum=0.0,epsilon=0.00001, centered=True)

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, 
    decay_steps=250000 // update_period, 
    end_learning_rate=0.01)

agent = DqnAgent(tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000, # <=> 32,000 ALE frames
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99, # discount factor
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step))

agent.initialize()


# In[173]:


from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000)

replay_buffer_observer = replay_buffer.add_batch


# In[174]:


from tf_agents.metrics import tf_metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
    ]


# In[175]:


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# In[176]:


from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)


# In[177]:


from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)


# In[178]:


from tf_agents.policies.random_tf_policy import RandomTFPolicy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(tf_env,
                                initial_collect_policy,
                                observers=[replay_buffer.add_batch],
                                num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()


# In[179]:


tf.random.set_seed(9) # chosen to show an example of trajectory at the end of an episode

#trajectories, buffer_info = replay_buffer.get_next( # get_next() is deprecated
#    sample_batch_size=2, num_steps=3)

trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps=3,
    single_deterministic_pass=False)))


# In[180]:



trajectories._fields


# In[181]:


trajectories.observation.shape


# In[182]:



from tf_agents.trajectories.trajectory import to_transition

time_steps, action_steps, next_time_steps = to_transition(trajectories)
time_steps.observation.shape


# In[183]:


trajectories.step_type.numpy()


# In[184]:


plt.figure(figsize=(10, 6.8))
for row in range(2):
    for col in range(3):
        plt.subplot(2, 3, row * 3 + col + 1)
        plot_observation(trajectories.observation[row, col].numpy())
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
save_fig("sub_episodes_plot")
plt.show()


# In[209]:


dataset = replay_buffer.as_dataset(
sample_batch_size=64,
num_steps=2,
num_parallel_calls=3).prefetch(3)


# In[202]:


from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)
checkpoint = tf.train.Checkpoint(agent)


# In[217]:


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
    


# In[218]:


train_agent(n_iterations=1000)


# In[219]:


import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)


# In[220]:


import PIL

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

image_path = os.path.join("images", "rl", "myAgentPlays.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=90,
                     loop=0)


# In[221]:


get_ipython().run_cell_magic('html', '', '<img src="images/rl/myAgentPlays.gif" />')


# In[215]:


checkpoint = tf.train.Checkpoint(agent)


# In[216]:


save_path = checkpoint.save('agent/lastModelCheckpoint')


# In[197]:


eval_policy = agent.policy


# In[198]:


saver = tf_agents.policies.PolicySaver(eval_policy, batch_size = None)


# In[200]:


saver.save("savedPolicy")


# In[ ]:




