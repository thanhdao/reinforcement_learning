# simple_plot.py
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# x = np.arange(0, math.pi*2, 0.05)
# y = np.sin(x)
# plt.plot(x, y)
# plt.xlabel('angle')
# plt.ylabel('sine')
# plt.title('sine wave')
# plt.show()

from numpy import *
from pylab import *

x = linspace(-3, 3, 30)
y = x ** 2
# plot(x, y, 'r')
plot(x, sin(x))
plot(x, cos(x), 'r-')
plot(x, -sin(x), 'g--')
show()