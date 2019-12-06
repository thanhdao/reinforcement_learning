# k_armed_bandit.py
import random
import numpy as np
import matplotlib.pyplot as plt

k_armed = range(10)
rewards = {}
time_steps = 1000
runs = 2000

# for k in k_armed:
#   rewards[k] = []
#   for t in range(time_steps):
#     time_step_rewards = []
#     for i in range(runs):

#       time_step_rewards.append(random.gauss(0, 1))
#     rewards[k].append(np.mean(time_step_rewards))

# print(rewards)
# epsilon:

# for t in range(time_steps)]

# k armed bandits
# rewards for each bandits generate by gaussian distribution N(0, 1)
# select a based on max rewards of all bandit

# calculate average reward after all
# rewards = {}

average_rewards = []
actions = []
for k in k_armed:
    rewards[k] = []



# for r in range(runs):
for i in range(time_steps):

  max = 0
  selected_k = 0
  for k in k_armed:
    reward = np.random.normal(0 ,1)
    rewards[k].append(reward)
    average = sum(rewards[k])/len(rewards[k])
    if max < reward:
      max = reward
      select_k = k

  average_rewards.append(max)
  actions.append(selected_k)

# for 

plt.plot(range(time_steps), average_rewards)
plt.show()




  