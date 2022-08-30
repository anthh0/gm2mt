import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Propagator import Propagator
from Ring import Ring
from StateGenerator import StateGenerator
from Kicker import Kicker

import auxiliary as aux
import numpy as np
import pathlib
import shutil
import os

#TODO: save simulation parameters in text file
#TODO: produce animation / don't produce option
#TODO: handle no muon survive case (e.g. b_k = 250)

m = 1.883531627E-28 # muon mass in kg
q = 1.602176634E-19 # muon charge in C
c = 299792458 # speed of light in m/s
B = 1.4513 # nom. field strength in T
T0 = 1.4914E-7


field_n = 0.1
r0s, p_avgs  = np.empty(shape = 13), np.empty(shape = 13)
for i in range(13):
    propagator = Propagator(
        # output = None,
        output = 'quad_test',
        display_options_s = [], # ["r", "phi", "dr", "dphi", "p", "animation"]
        display_options_m = [], #  ["p", "f", "hist2d", "delta_p", "spatial"]
        compare_dir = None,
        animate = False,
        store_root = False
    )

    # method = 'verlet-cart'
    # method = 'verlet-cyl'
    method = 'rk4-cyl'

    generator = StateGenerator(seed = None)
    r0 = aux.r_magic - 0.030 + (i * 0.005)
    init_state = generator.generate(
        # muons = 1E6,
        muons = 1,
        method = method,
        # r = dist.Gaussian(mean = aux.r_inj, std = 3.5E-3),
        # r = aux.r_inj,
        r = r0,
        momentum = aux.p_magic * 1.01,
        # momentum = dist.Gaussian(mean = aux.p_magic, std = 0.008 * aux.p_magic),
        # offset = 30,
        # offset = 43,
        alpha = 0,
        # alpha = dist.Gaussian(mean = 0, std = 2.6E-3),
        # alpha = dist.Gaussian(mean = 2.5E-3, std = 2.6E-3),
        phi_0 = 0,
        # t = 100
        t = 0
        # t = dist.Custom(dir = "pulseshape_1_12543_mod", zero = 24750) # originally 24790
    )


    def_ring = Ring(
        r_max = 7.157, # radius of outer ring edge (m)
        r_min = 7.067, # radius of inner ring edge (m)
        k_i = aux.k1_i, # angle where kicker region begins, measured downstream from injector (degrees)
        k_f = aux.k3_f, # angle where kicker region ends
        b_nom = aux.B, # Nominal b-field strength (Tesla)
        # b_k = Kicker("file", "traces-Ir", b_norm = 204), # Kicker b-field strength (Gauss)
        # b_k = Kicker("uniform", 200),
        b_k = Kicker("uniform", 0),
        # b_k = Kicker("one turn", 204),
        quad_model = "linear", # "linear" "full"
        quad_num = 1, # 1 or 4
        n = field_n,
        # n = 0.108, # Equivalent to the electric field "spring constant" k
        fringe = False # turn fringe effect on or off
    )

    r_co_t, r_co_phi, avg_p, T = propagator.propagate(
        init_state = init_state, # a muon_num x 5 NumPy array containing the initial conditions of the muons
        ring = def_ring,
        state_generator = generator,
        integration_method = method,
        exit_check = True,
        dt = 0.1, # the time-step in nanoseconds
        t_f = 1000, # end time in nanoseconds
    )
    print(r_co_t, r_co_phi, avg_p, T)
    r_co_p = aux.r_magic / (1-field_n) * (avg_p - aux.p_magic) / aux.p_magic * 1000

    # r_co_ts[i] = r_co_t
    # r_co_phis[i] = r_co_phi
    # r_co_ps[i] = r_co_p
    # Ts[i] = T
    # ns[i] = field_n




# n = np.array([0,   0.02,  0.04,  0.06,  0.08,  0.10,  0.12,  0.14])
# n2 = np.array([0,   0.1,  0.2,  0.3,  0.4])
# T = np.array([149, 150.5, 151.9, 153.2, 154.5, 155.7, 156.8, 157.8])
# r_co = np.array([3.736921883046307E-08, -0.32267387317608254, -0.646842910666301, -0.9714947050381006, -1.2956125072354219, -1.6181951973632813, -1.9382782349479655, -2.2549504203976944])
# r_co2 = np.array([7.1435, 5.9271, 4.3949, 2.61221, 0.76708])
T_frac = [number / 149.14 for number in Ts]

xvals = np.linspace(0, 0.19, 50)
yvals = 1 / np.sqrt(1 - xvals)
# print(q * B * T0 / 2 / np.pi / m)
# yvals = 1 / np.sqrt(1 - (q*n*B*T0 / 2 / np.pi / m) )

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlabel("field index")
ax1.set_ylabel("T / T_magic")
ax1.plot(xvals, yvals, label = '1/sqrt(1 - n)')
# ax1.plot(n, T_frac, label = 'simulation')
ax1.errorbar(ns, T_frac, yerr = 6E-4, capsize = 1, label = 'simulation')
ax1.legend()

ax2.set_xlabel("field index")
ax2.set_ylabel("R_co (mm)")
ax2.plot(ns, r_co_ts, label = "avg r over t")
ax2.plot(ns, r_co_phis, label = "avg r over phi")
ax2.plot(ns, r_co_ps, label = "r_co, avg_p")
ax2.legend()

dir = 'results/quad_testing'
if pathlib.Path(dir).is_dir():
    shutil.rmtree(dir)
os.mkdir(dir)
pdf = PdfPages("results/quad_testing/plots.pdf")
pdf.savefig(fig)
pdf.close()

plt.show()