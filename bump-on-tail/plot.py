from pylab import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

f = open("out.dat")

t = f.readline().split(',');

Nx, Nv, Lx, Lv = int(t[0]), int(t[1]), float(t[2]), float(t[3])

A = np.loadtxt("out.dat", skiprows=1, delimiter=',')

X = np.linspace( 0., Lx, Nx)
V = np.linspace(-Lv, Lv, Nv)


# Make data.
X = range(Nx)#np.arange(-5, 5, 0.25)
Y = range(Nv)#np.arange(-5, 5, 0.25)

fig = plt.figure()

fig = plt.figure()
ax = fig.gca(projection='3d')

X, V = np.meshgrid(X, V)

# Plot the surface.
surf = ax.plot_surface(X, V, A.T, cmap=cm.coolwarm, linewidth=1, antialiased=True, shade=True)


# Customize the z axis.

# Add a color bar which maps values to colors.
"""
fig.colorbar(surf, shrink=0.5, aspect=5)

contourf(V,X,A)
xlabel("V")
ylabel("X")

"""
plt.show()
