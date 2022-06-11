######################################
# Implemented the code using stable baselines from examples

from pyvirtualdisplay import Display
import gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# creating env
env = gym.make('LunarLander-v2')
env = Monitor(env, log_dir)

# Agent
dqn_model = DQN('MlpPolicy', env, verbose=1, batch_size=128, buffer_size=50000, exploration_final_eps=0.1, exploration_fraction=0.12,
                gamma=0.99, gradient_steps=-1, learning_rate=0.00063, learning_starts=0, policy_kwargs=dict(net_arch=[256, 256]),
                target_update_interval=250, train_freq=4)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# # Training the agent
timesteps = 1e5
dqn_model.learn(total_timesteps=int(timesteps), callback=callback)

# # plot the training reward
plot_results([log_dir], timesteps,
             results_plotter.X_TIMESTEPS, "DQN LunarLander")
plt.savefig('RewardsVsTime_DQN_Lunar_tuned1.png')
plt.show()

# Saving the agent for evaluation
dqn_model.save("dqn_lunar_tuned1")

# delete trained model
del dqn_model

# Load the trained agent
model = DQN.load("dqn_lunar_tuned1", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

display = Display(visible=0, size=(400, 300))
display.start()

# Enjoy trained agent
obs = env.reset()
prev_screen = env.render(mode='rgb_array')
plt.imshow(prev_screen)

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"rewards: {rewards}")
    screen = env.render(mode='rgb_array')
    plt.savefig('dqn_lunar_tuned_render.png')
    plt.imshow(screen)
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())

ipythondisplay.clear_output(wait=True)
env.close()
