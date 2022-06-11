from pyvirtualdisplay import Display
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
import glob
from IPython import display
import base64
import io
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from collections import deque, namedtuple

# Create Env
env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


class DQN_Network(nn.Module):

    def __init__(self, state_size, action_size, seed):

        super(DQN_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):

        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x


Buff_size = int(1e5)  # replay buffer size
Batch_size = 64       # minibatch size
gamma = 0.99          # discount factor
tau = 1e-3            # for soft update of target parameters
lr = 5e-4             # learning rate
update = 4            # how often to update the network
bias = 10


class Agent():

    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.dqnetwork_local = DQN_Network(state_size, action_size, seed)
        self.dqnetwork_target = DQN_Network(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.dqnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, Buff_size, Batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % update

        if (self.time_step == 0):
            if(len(self.memory) > Batch_size):
                sample_space = self.memory.sample()
                self.learn(sample_space, gamma)

    def action_(self, state, epsilon=0):

        state = torch.from_numpy(state).float().unsqueeze(0)
        self.dqnetwork_local.eval()

        with torch.no_grad():
            action_values = self.dqnetwork_local(state)
        self.dqnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            q_targets_next = self.dqnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1) - bias

            q_targets = rewards + (q_targets_next * gamma) * (1 - dones)

            # q_targets = rewards + ((1 - gamma) * bias +
            #                        q_targets_next * gamma) * (1 - dones)
            # print(q_targets)

            q_targets = q_targets + bias

        q_expected = self.dqnetwork_local(states).gather(1, actions)

        # minimize loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update
        for target_param, local_param in zip(self.dqnetwork_target.parameters(), self.dqnetwork_local.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):

        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 0  # eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.action_(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        # eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.dqnetwork_local.state_dict(),
                       'checkpoint_bias10_0eps.pth')
            break
    return scores


agent = Agent(state_size=8, action_size=4, seed=0)
scores = dqn()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("Dqn_lunar_bias10_0eps.png")
plt.show()


# For visualization


def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def show_video_of_model(agent, env_name):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(
        env, path="video/{}__bias10_0eps.mp4".format(env_name))
    agent.dqnetwork_local.load_state_dict(
        torch.load('checkpoint_bias10_0eps.pth'))

    display = Display(visible=0, size=(400, 300))
    display.start()

    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')

        vid.capture_frame()

        action = agent.action_(state)

        state, reward, done, _ = env.step(action)
    env.close()


agent = Agent(state_size=8, action_size=4, seed=0)
show_video_of_model(agent, 'LunarLander-v2')

show_video('LunarLander-v2')
