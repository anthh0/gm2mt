from gm2mt.Propagator import Propagator
from gm2mt.Ring import Ring
from gm2mt.StateGenerator import StateGenerator
from gm2mt.Kicker import Kicker
import gm2mt.Distributions as dist
import gm2mt.auxiliary as aux

import numpy as np
import numba as nb
import os

nice_level = 5                                                                          # idk why this is set this way specifically #
os.nice(nice_level)

# threads = 10
# nb.set_num_threads(threads)
# print(nb.get_num_threads())

propagator = Propagator(
    # output = None,
    output = 'x_alpha_testing',
    plot = True,
    display_options_s = [],                                                             # ["r", "phi", "dr", "dphi", "p", "animation"]
    display_options_m = [],                                                             # ["p", "f", "hist2d", "delta_p", "spatial"]
    # p_acceptance_dir = "n0k1",
    p_acceptance_dir = None,
    plot_option = "presentation",                                                       # "presentation" or "maximize_graph"
    animate = False,
    store_root = True,
    suppress_print = False,     
    store_f_dist = False,
    multiplex_plots = ['mt1', 'x_alpha']                                                # multi muon multiplex sims: ["p_dist", "f", "mt3", "mt1", "pulse", "spatial", "inj", "x_alpha", "stats"]
    # multiplex_plots = ['trajectory']                                                    # single muon multiplex sims: ["trajectory", "offset", "phi", "vr", "vphi", "p", "kicker"]
)

ring = Ring(
    r_max = aux.r_magic + 0.045,                                                        # radius of the ring is from 7.067 to 7.157 [m]
    r_min = aux.r_magic - 0.045,
    b_nom = aux.B_nom,                                                                  # Nominal b-field strength (T)
    b_k = Kicker("file", "KickerPulse_Run2_June19_Normalized", b_norm = 204, kick_max = 100, kicker_num = 3),
    quad_model = "linear",                                                              # "linear", "full_interior" or "full_exterior"
    quad_num = 4,                                                                       # 1 or 4
    n = 0.108,                                                         
    collimators = "discrete",                                                           # "discrete", "continuous", or "none" --> fails to run
    fringe = False
)

generator = StateGenerator(
    mode = "mc",                                                                        # 'mc' or 'bmad'
    seed = None,
    muons = 5E5,
    momentum = dist.Gaussian(mean = 100, std = [0.6, 0.8, 1.0]),
    x_alpha = dist.Custom2D(dir = "/x_alpha_profiles/PostInflectorFull_RubinClean"),
    # initial_offset = dist.Gaussian(mean = aux.r_inj_offset - 2.0, std = 3.2),
    # alpha = dist.Gaussian(mean = 0.3, std = 2.5),
    t = dist.Custom(dir = "/pulseshape_run2_ns/pulseshape_0_run2_ns", zero = 56.5)
)

propagator.propagate(
    ring = ring,
    state_generator = generator,
    integration_method = 'rk4',                                                         # 'rk4' or 'rkn' (not quite ready yet)
    dt = 0.1,                                                                           # the time-step in nanoseconds
    t_f = 4000                                                                          # end time in nanoseconds
)