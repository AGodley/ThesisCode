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

# Figure 1
# Measurement angle plot
fig1, ax1 = plt.subplots()
ax1 = plt.subplot(projection='polar')
# n, changed to plot part of the sample trajectory
n = 25

# Sets up radii for the polar plot
r = np.linspace(1, 10, n)
# Polar plot
ax1.plot(angles[0:n], r, 'o', ms=5)

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
    ax1.plot(np.append(arc, angles[i+1]), np.append(r[i]*np.ones(points), r[i+1]), 'r--', linewidth=2)
# Positions for theta labels
positions = pi/180*np.arange(0, 360, 30)
ax1.set_xticks(positions)
# Corresponding labels
labels = [str(i)+'$^\circ$' for i in np.arange(0, 360, 30)]
ax1.set_xticklabels(labels)
# Adds space between graph and theta labels to stop overlap
ax1.xaxis.set_tick_params(pad=12)
# Removes r labels
ax1.set_yticklabels([])
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Layout
plt.tight_layout()
# plt.show()

# Figure 2
# Linear measurement angle plot
fig2, ax2 = plt.subplots(frameon=False)
# Angle shift, chosen to take out large jumps
shift = 5*pi/6
# Plot
ax2.plot(np.arange(len(angles)), -np.angle(np.exp(1j*(angles+shift))), linewidth=2)
# Adjustments
ax2.set_yticks([0, -pi/6, -2*pi/6, -3*pi/6, -4*pi/6, -5*pi/6, -pi])
labels2 = ['']
for i in ['180', '135', '90', '45', '0']:
    labels2.append(i+'$^\circ$')
labels2.append('')
ax2.set_yticklabels(labels2, fontsize=12)
ax2.set_xlabel('Trajectory length, n', fontsize=14)
ax2.set_ylabel('Measurement angle', fontsize=14)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()
# Tick fontsize
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Alternative to figure 1
fig3, ax3 = plt.subplots()
# Plots a circle
angle = np.linspace(0, pi)
radius = 10
circ_x = radius * np.cos(angle)
circ_y = radius * np.sin(angle)
bottom = np.linspace(-10, 10)
ax3.plot(circ_x, circ_y, 'k', linewidth=1)
ax3.plot(bottom, np.zeros(len(bottom)), 'k', linewidth=1)
ax3.axis('equal')
# Plots the trajectory in this circle
angles += 5*pi/6
radii = np.linspace(1, 10, len(angles))
#
ax3.plot(radii*np.sin(angles), radii*np.cos(angles))
plt.show()