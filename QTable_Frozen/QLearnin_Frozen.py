import gym
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})


# Create Env
env = gym.make('FrozenLake-v0', is_slippery=False)
env.seed(0)
num_states = env.observation_space.n
num_actions = env.action_space.n
print('State shape: ', num_states)
print('Number of actions: ', num_actions)
env.reset()
env.render()

# creating an empty Q table
q_table = np.zeros((num_states, num_actions))


gamma = 0.99         # discount factor
lr = 5e-4            # learning rate
update = 4            # how often to update the network
bias = -50


class Agent():

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

    def action_(self, state, epsilon=0):

        if random.random() > epsilon:
            return np.argmax(q_table[state, :])
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, reward, state, new_state, action, gamma):

        # states, actions, rewards, next_states, dones = experiences

        if(state == 15):
            Q_learn = reward - bias*(1-gamma) - q_table[state, action]

        else:
            Q_learn = reward - bias*(1-gamma) + gamma * \
                np.max(q_table[new_state]) - q_table[state, action]

        q_table[state, action] = q_table[state, action] + lr * (Q_learn)


def Q_learning(n_episodes=20000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    print("Qtable at the start:", q_table)
    scores = []                        # list containing scores from each episode
    # scores_window = deque(maxlen=100)  # last 100 scores
    eps = 0  # eps_start                    # initialize epsilon
    outcomes = []
    counter = []
    s = []
    q_initialstate = []
    t_count = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        score_ = 0
        outcomes.append(0)
        for t in range(max_t):
            q_initialstate.append(q_table[0][0])
            action = agent.action_(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.learn(reward, state, next_state, action, gamma)
            state = next_state
            t_count += 1
            score_ += reward
            if done:
                outcomes[-1] = 1
                score += reward
                break
            score += -0.01
        # scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        counter.append(t_count)
        s.append(score_)
        # eps = max(eps_end, eps_decay*eps)  # decrease epsilon

    return scores, outcomes, q_table, counter, q_initialstate, s


agent = Agent(state_size=16, action_size=4)
scores, outcomes, qtable, counter, q_values, score_ = Q_learning()

print("Qtable at the end:", qtable)
print(sum(scores)/20000)
# plt.plot(np.cumsum(scores)/np.arange(1, 20000+1), '.')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.cumsum(score_)/np.arange(1, 20000+1), '.')

# plt.plot(np.array(counter)/100, scores)
plt.ylabel('Score')
plt.xlabel('Time Steps $(10^2)$')
plt.title("Learning Curve")
plt.savefig("Learningrate_frozen_.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(q_values))/100, q_values)
plt.ylabel('Q_values')
plt.xlabel('Time Steps $(10^2)$')
plt.title("Q values of state 0")
plt.savefig("Qvalue_frozen_.png")
plt.show()


# For evaluation
env.reset()

for episode in range(10):
    state = env.reset()
    step = 0
    done = False
    print("EPISODE ", episode)

    for step in range(100):

        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        if done:

            env.render()
            if new_state == 15:
                print("We reached our Goal")
            else:
                print("We fell into a hole")

            print("Number of steps", step)

            break
        state = new_state
env.close()
