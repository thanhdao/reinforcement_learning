# k_armed_bandit.py
import random
import numpy as np
import matplotlib.pyplot as plt

'''

Problem

k options
each choice get 1 reward form a stationary probability distribution
maximize the expected total reward over some time period, 1000 example 

Example: 
10 options
Choose 1 k at every step
Random
choose k1 get r1
choose k2 get r2
...
choose k1000 get r1000

E(total)



Using q value function to choose k

calculate q value function of all k at every step

Greedy
Choose k with the max q



'''

def q_function(action):

  return value


def get_action(q_scores):
  action = -1
  value = -1
  for act, val in q_scores.items():
    if value < val:
      action = act


  return action

runs = 1

time_steps = 1000
k_arms = range(10)
epsilon = 0

rewards_by_actions ={}
expected_rewards = []
q_scores = {}

for k in k_arms:
  q_scores[k] = np.random.rand()
  rewards_by_actions[k] = []

for r in range(runs):
  rewards = []
  for i in range(time_steps):
    
    reward = np.random.rand()
    if reward < epsilon:
      action = np.random.choice(k_arms)
    else:
      action = get_action(q_scores)
    rewards_by_actions[action].append(reward)
    q_scores[action] = sum(rewards_by_actions[action]) / len(rewards_by_actions[action])
    expected_rewards.append(q_scores[action])

# print('Actions values: ', rewards_by_actions)
# print(expected_rewards)
figure = plt.figure(figsize=(10,10))
ax1 = figure.add_axes([0,0,0.5,0.5])
plt.yticks(range(-3,3))
plt.xticks(range(1000))
ax1.plot(range(time_steps), expected_rewards)




ax2 = figure.add_axes([0,0.5,0.5,0.5])
all_data = [[random.gauss(0,1) for j in range(2000)] for k in range(1, 11)]
  
# plot violin plot
ax2.violinplot(all_data,showextrema=True,
                   showmeans=True,
                   showmedians=False)
plt.title('Violin plot')
plt.yticks(range(-3,3))
plt.xticks(range(1,11))

plt.show()

# 10-armed test bed
# data_len = 2000
# for i in range(data_len):

