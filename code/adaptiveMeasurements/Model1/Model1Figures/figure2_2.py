# Alfie Godley
# 14/02/2022
# Generating the second figure

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from math import sqrt, pi

data = pd.read_csv('../../../data/Model1/fig2_measurement_angles.csv', header=None)
angles = data.loc[0].to_numpy()

# Measurement angle plot
fig0 = plt.figure()
ax0 = plt.subplot(1, 1, 1, projection='polar')
ax0.set_title('Measurement Angle for case {}'.format(2), fontsize=18)
# n, can be changed to just plot part of the sample trajectory
n = 100
# Sets up radii for the polar plot
r = np.linspace(2, 10, n)
# Polar plot
ax0.plot(angles[0:n], r, 'o', ms=4)
# Arcs between points
for i in np.arange(n-1):
    points = 100
    if abs(angles[i+1] - angles[i]) < pi:
        arc = np.linspace(angles[i], angles[i+1], points)
    else:
        if angles[i] < pi:
            arc0 = np.linspace(angles[i], 0, int(np.ceil(0.5*points)))
            arc1 = np.linspace(2*pi, angles[i+1], int(np.floor(0.5*points)))
            arc = np.append(arc0, arc1)
        else:
            arc0 = np.linspace(angles[i], 2*pi, int(np.ceil(0.5*points)))
            arc1 = np.linspace(0, angles[i+1], int(np.floor(0.5*points)))
            arc = np.append(arc0, arc1)
    ax0.plot(np.append(arc, angles[i+1]), np.append(r[i]*np.ones(points), r[i+1]), 'r--', linewidth=1)
ax0.set_xticks(pi/180*np.arange(0, 360, 30))
# ax0.set_xticklabels(0.5*np.arange(0, 360, 30))
ax0.set_yticklabels([])
plt.show()

# Linear measurement angle plot
fig1, ax1 = plt.subplots(1, 1)
shift = 5*pi/6
ax1.plot(np.arange(len(angles)), np.angle(np.exp(1j*(angles+shift))))
ax1.set_yticks([0, pi])
ax1.set_yticklabels(['$\\frac{-5\pi}{6}$', '$\\frac{\pi}{6}$'], fontsize=14)
ax1.set_xlabel('n', fontsize=14)
ax1.set_ylabel('$\phi$', fontsize=14)
plt.show()
