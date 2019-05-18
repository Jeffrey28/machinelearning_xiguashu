#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @description 
# @file 
# @author 
# @date 
# @version

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# x = np.array([1, 2])
# y = np.array([1, 2])
# z = np.array([[0, 1], [-1, 0]])
x = np.linspace(1, 2, 100)
y = np.linspace(1, 2, 100)
X, Y = np.meshgrid(x, y)
z = np.sign(X-Y)
plt.xlim(1, 2)
plt.ylim(1, 2)
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(z))])
plt.contourf(X, Y, z, cmap=cmap)
plt.show()
