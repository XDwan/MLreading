# -*- coding: utf-8 -*-

"""
@Time : 2021/3/24
@Author : XDwan
@File : Simple_Neural_Network
@Description : 
"""
import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(0., 1., 0.001)
x2 = np.arange(0., 1., 0.001)


def F(x1, x2):
    if x1 > 0.5 and x2 > 0.5:
        return 1
    return 0


X1, X2 = np.meshgrid(x1, x2)

f = F(X1, X2)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X1, X2, f, cmap='viridis')
plt.show()
