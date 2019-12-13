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
********************** Choose Random ***************************************
choose k1 get r1
choose k2 get r2
...
choose k1000 get r1000

E(total)
'''

'''
********************** Using estimate function ***************************
Using q value function to choose k

calculate q value function of all k at every step
Greedy
Choose k with the max q
'''

def q_function(action):

  return value

def get_action(scores):
  action = -1
  value = -1
  for act, val in scores.items():
    if value < val:
      action = act

  return action

runs = 1
time_steps = 1000
k_arms = range(10)
epsilon = 0

rewards_by_actions = {}
expected_rewards = []
estimate_scores = np.zeros(10)

actions_values = np.random.randn(10)

for k in k_arms:
  rewards_by_actions[k] = []

for r in range(runs):

  for i in range(time_steps):
    

    nongreedy = np.random.rand()
    if nongreedy < epsilon:
      action = np.random.choice(k_arms) # Choose random
    else:
      action = np.argmax(estimate_scores) # Choose using action

    reward = np.random.randn() + actions_values[action] # distribution of each bandit is gaussian(action value, 1)
    rewards_by_actions[action].append(reward)
    estimate_scores[action] = sum(rewards_by_actions[action]) / len(rewards_by_actions[action])
    expected_rewards.append(estimate_scores[action])

figure = plt.figure(figsize=(10,5))

# 10 armed test bed
plt.subplot(121)
all_data = np.random.randn(2000, 10) + np.random.randn(10)

plt.violinplot(all_data,showextrema=False, showmeans=True, showmedians=False)                                     
plt.title('10-armed test bed')
# plt.yticks(range(-3,4))
plt.xticks(range(1,11))
plt.xlabel('Action')
plt.ylabel('Reward distribution')

# Action values
plt.subplot(122)
# plt.yticks(range(-3,3,1))
plt.xticks(range(0,1000,100))
plt.title('Action values')
plt.plot(range(time_steps), expected_rewards)                                                                                                             


plt.show()


