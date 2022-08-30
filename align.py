import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import ROOT

import gm2mt.auxiliary as aux
import gm2mt.Distributions as dist
from gm2mt.Ring import Ring, _MRing
from gm2mt.Plotter import Plotter, AlignPlotter
from gm2mt.Propagator import Propagator
from gm2mt.Kicker import Kicker
from gm2mt.StateGenerator import StateGenerator

# Column 4 - radial displacement  (x)
# Column 5-  angle
# Column 6,7 are vertical displacement and angle
# column 9 - fractional momentum offset (with respect to magic momentum)
# column 10 - time
propagator = Propagator(
    output = 'test2',
    plot = False,
    display_options_s = [], # ["r", "phi", "dr", "dphi", "p", "animation"]
    display_options_m = [], #  ["p", "f", "hist2d", "delta_p", "spatial"]
    p_acceptance_dir = None,
    plot_option = "presentation",
    animate = False,
    store_root = True,
    suppress_print = False,
    multiplex_plots = []
)

ring = Ring(
    r_max = aux.r_magic + 0.045,
    r_min = aux.r_magic - 0.045,
    b_nom = aux.B_nom, # Nominal b-field strength (T)
    b_k = Kicker("file", "KickerPulse_Run2_June19_Normalized", b_norm = 204, kick_max = 100, kicker_num = 3),
    quad_model = "linear", # "linear" "full"
    quad_num = 4, # 1 or 4
    n = 0.108,
    collimators = "discrete",
    fringe = False # turn fringe effect on or off
)

generator = StateGenerator(
    mode = "mc",
    seed = None,
    muons = 5E5,
    initial_offset = dist.Gaussian(mean = aux.r_inj_offset - 2.6, std = 4.2),
    momentum = dist.Gaussian(mean = 100, std = 0.8),
    alpha = dist.Gaussian(mean = -0.9, std = 3.1),
    phi_0 = 0,
    t = dist.Custom(dir = "pulseshape_run2_ns/pulseshape_0_run2_ns", zero = 60)
    # t = dist.Custom(dir = "pulseshape_run2_ns/pulseshape_avg_run2_ns", zero = 60)
)

propagator.align(
    state_generator = generator,
    ring = ring,
    integration_method = "rk4",
    dt = 0.1, # the time-step in nanoseconds
    t_f = 1000, # ending time in ns
    t_0 = 100,
    coarseRange = 100,
    coarseStep = 10,
    fineRange = 10,
    fineStep = 1
)