# k_armed_bandit.py
import random
import numpy as np

k_armed = range(10)
rewards = {}
time_steps = 1000
runs = 2000

for k in k_armed:
  rewards[k] = []
  for t in range(time_steps):
    time_step_rewards = []
    for i in range(runs):

      time_step_rewards.append(random.gauss(0, 1))
    rewards[k].append(np.mean(time_step_rewards))

print(rewards)

  