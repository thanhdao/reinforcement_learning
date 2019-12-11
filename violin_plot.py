# violin_plot.py
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import trange

# fig, axes = plt.subplots( figsize=(9, 4))

# Fixing random state for reproducibility
# np.random.seed(19680801)
# 10 armed test bed

runs = 2000
time_steps = 1000

# generate some random test data
# all_data = [[np.mean([random.random() for i in range(runs)]) * 40 for j in range(time_steps)] for k in range(1, 11)]
# all_data = [[np.random.normal(0 ,1) for j in range(runs)] for k in range(1, 11)]
all_data = [[np.random.randn() + np.random.randn() for j in range(runs)] for k in range(1, 11)]
all_data = all_data + np.random.randn(1,10)
# plot violin plot
plt.figure(figsize=(10, 10))
plt.violinplot(all_data,showextrema=True,
                   showmeans=True,
                   showmedians=False)
plt.title('Violin plot')
plt.yticks(range(-3,3))
plt.xticks(range(1,11))


# plot box plot

plt.show()


