from Propagator import Propagator
from Ring import Ring
from StateGenerator import StateGenerator
from Kicker import Kicker
from Plotter import Plotter
import Distributions as dist
import matplotlib.pyplot as plt

import auxiliary as aux
import numpy as np
dir = "s2-two_kick"
# dir = input("Directory: ")
t, upper, lower = np.loadtxt("windows/windows_s2-two_kick.txt", skiprows = 1, unpack = True)
# t-= 37.5
results = np.load(f"results/{dir}/final_states.npz")
inj = np.load(f"results/{dir}/init_states.npz")['t'] * 10**9
lost = results['lost']
r = results['r']
vr = results['vr']
vphi = results['vphi']
offset = aux.p_to_rco(aux.state_to_mom_cyl(r, vr, vphi), n = 0.108)

offset_s, inj_s = Plotter._mask_lost(lost, offset, inj)

fig, ax = plt.subplots()

# hist, xedges, yedges, image = ax.hist2d(inj_s, offset_s, range = [[0, 200], [-15, 45]], cmap = 'jet', bins = (200, 150))
hist, xedges, yedges, image = ax.hist2d(inj_s, offset_s, cmap = 'jet', bins = (200, 150))

xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
maxima = np.empty(shape = len(xcenters))
minima = np.empty(shape = len(xcenters))

for i in range(len(maxima)):
    maxima[i] = ycenters[np.nonzero(hist[i,:])[0][-1]]
    minima[i] = ycenters[np.nonzero(hist[i,:])[0][0]]
ax.plot(t, upper, color = 'white', linewidth = 1)
ax.plot(t, lower, color = 'purple', linewidth = 1)
plt.show()

# array = np.empty(shape = (len(maxima), 3))
# array[:,0] = xcenters
# array[:,1] = maxima
# array[:,2] = minima

# with open('windows.txt', "w") as f:
# np.savetxt('windows.txt', array, fmt = "%6.4f", header = "injection (ns) | upper bound (mm) | lower bound (mm)")