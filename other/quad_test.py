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

m = 1.883531627E-28 # muon mass in kg
q = 1.602176634E-19 # muon charge in C
c = 299792458 # speed of light in m/s
B = 1.4513 # nom. field strength in T
T0 = 1.4914E-7

prefactor = 1

T_phis, r_co_t_ints, r_co_phis, r_co_ps, Ts, ns = np.empty(shape = 20), np.empty(shape = 20), np.empty(shape = 20), np.empty(shape = 20), np.empty(shape = 20), np.empty(shape = 20)
for i in range(20):
    field_n = i * 0.01
    propagator = Propagator(
        # output = None,
        output = 'quad_test',
        display_options_s = [], # ["r", "phi", "dr", "dphi", "p", "animation"]
        display_options_m = [], #  ["p", "f", "hist2d", "delta_p", "spatial"]
        animate = False,
        store_root = False
    )

    # method = 'verlet-cart'
    # method = 'verlet-cyl'
    method = 'rk4-cyl'

    generator = StateGenerator(seed = None)
    init_state = generator.generate(
        # muons = 1E6,
        muons = 1,
        method = method,
        # r = dist.Gaussian(mean = aux.r_inj, std = 3.5E-3),
        # r = aux.r_inj,
        r = aux.r_magic + 0.020,
        momentum = aux.p_magic * prefactor,
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
        b_nom = aux.B_nom, # Nominal b-field strength (Tesla)
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

    T_phi, r_co_t_int, r_co_phi, avg_p, T = propagator.propagate(
        init_state = init_state, # a muon_num x 5 NumPy array containing the initial conditions of the muons
        ring = def_ring,
        state_generator = generator,
        integration_method = method,
        dt = 0.01, # the time-step in nanoseconds
        t_f = 1000, # end time in nanoseconds
    )
    print(T_phi, r_co_t_int, r_co_phi, avg_p, T)
    r_co_p = aux.r_magic / (1-field_n) * (avg_p - aux.p_magic) / aux.p_magic * 1000

    T_phis[i] = T_phi
    r_co_t_ints[i] = r_co_t_int
    r_co_phis[i] = r_co_phi
    r_co_ps[i] = r_co_p
    Ts[i] = T
    ns[i] = field_n

T_frac = Ts / 149.14

xvals = np.linspace(0, 0.19, 20)
yvals = 1 / np.sqrt(1 - xvals) / prefactor
T_error = (T_frac - yvals) / yvals

yvals2 = 2 * np.pi * aux.r_magic / np.sqrt(1 - xvals)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlabel("field index")
ax1.set_ylabel("Fractional error")
# ax1.plot(xvals, yvals, label = r'$1/\sqrt{1 - n}$', linewidth = 1)
ax1.plot(xvals, T_error, label = r'$\delta_T$', linewidth = 1)
# ax1.errorbar(ns, T_frac, yerr = 6E-4, capsize = 3, label = 'simulation', linewidth = 1)
ax1.legend()

r_t_error = (r_co_t_ints - r_co_ps) / (r_co_ps + (aux.r_magic*1000))
r_phi_error = (r_co_phis - r_co_ps) / (r_co_ps + (aux.r_magic*1000))

ax2.set_xlabel("field index")
ax2.set_ylabel("Fractional error")
# ax2.plot(ns, r_co_t_means, label = "avg r over t (mean method)")
ax2.plot(ns, r_t_error, label = r"$\delta_{r(t)}$")
ax2.plot(ns, r_phi_error, label = r"$\delta_{r(\phi)}$")
# ax2.plot(ns, r_co_t_ints, label = "avg r over t (int method)")
# ax2.plot(ns, r_co_phis, label = "avg r over phi")
# ax2.plot(ns, r_co_ps, label = "r_co, avg_p")
ax2.legend()

# ax3.set_xlabel("field index")
# ax3.set_ylabel("T_phi / T_phi_magic")
# ax3.plot(xvals, yvals2, label = '2pir_0/sqrt(1 - n)', linewidth = 1)
# ax3.plot(ns, T_phis, label = 'simulation', linewidth = 1)
# ax3.legend()

dir = 'results/quad_testing'
pdf = PdfPages("results/quad_testing/plots.pdf")
pdf.savefig(fig)
pdf.close()

plt.show()