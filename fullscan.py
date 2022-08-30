from Propagator import Propagator
from Ring import Ring
from StateGenerator import StateGenerator
from Kicker import Kicker
import Distributions as dist

import auxiliary as aux
import numpy as np
import pathlib
import shutil
import os

os.nice(5)

bnorms = np.linspace(201, 207, 7)
xmeans = aux.r_inj_offset - np.linspace(1.1, 4.1, 11)
xstds = np.linspace(2, 6.4, 12)
# pmeans = np.linspace(99.5, 100.5, 11)
# pstds = np.linspace(0.4, 1.2, 9)

alphastds = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]
alphastds = [2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
alphastds = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
alphameans = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6]
# tnorms = np.linspace(24805, 24850, 13)


xstds = np.linspace(3, 5.4, 5)
alphastds = np.linspace(2.4, 3.8, 5)
pmeans = np.linspace(99.7, 100.3, 5)
tnorms = np.linspace(24805, 24850, 5)

dir = 'fullscanLarge'
full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + dir
results_path = str(pathlib.Path(__file__).parent.absolute()) + "/results"

if pathlib.Path(full_path).is_dir():
    print(f"Overwriting directory /{dir}...")
    shutil.rmtree(full_path)
else:
    print(f"Creating directory /{dir}...")
os.mkdir(full_path)

for xstd_idx, xstd in enumerate(xstds):
    for alphastd_idx, alphastd in enumerate(alphastds):
        for pmean_idx, pmean in enumerate(pmeans):
            for tnorm_idx, tnorm in enumerate(tnorms):
                propagator = Propagator(
                    # output = None,
                    output = f'{dir}_{xstd_idx+1}_{alphastd_idx+1}_{pmean_idx + 1}_{tnorm_idx+1}',
                    plot = False,
                    display_options_s = [], # ["r", "phi", "dr", "dphi", "p", "animation"]
                    display_options_m = [], #  ["p", "f", "hist2d", "delta_p", "spatial"]
                    # p_acceptance_dir = "n0k1",
                    p_acceptance_dir = None,
                    plot_option = "presentation",
                    # plot_option = "maximize_graph",
                    animate = False,
                    # store_root = False,
                    store_root = True,
                    store_npz = False,
                    suppress_print = False,
                    store_f_dist = True,
                    multiplex_plots = ['mt1']
                    # multiplex_plots = ["mt1", "inj", "stats"]
                )

                ring = Ring(
                    # r_max = 7.157, # radius of outer ring edge (m)
                    # r_min = 7.067, # radius of inner ring edge (m)
                    r_max = aux.r_magic + 0.043,
                    r_min = aux.r_magic - 0.043,
                    b_nom = aux.B_nom, # Nominal b-field strength (T)
                    # b_k = Kicker("file", "traces-Ir", b_norm = list(np.linspace(180, 225, 2)), kick_max = 100, kicker_num = 3),
                    # b_k = Kicker("file", "traces-Ir", b_norm = np.linspace(180, 225, 2), kick_max = 100, kicker_num = 3),
                    # b_k = Kicker("file", "traces-Ir", b_norm = 204, kick_max = 1, kicker_num = 3),
                    b_k = Kicker("file", "traces-Ir", b_norm = 204, kick_max = 100, kicker_num = 3),
                    # b_k = Kicker("uniform", 100),
                    # b_k = Kicker("uniform", 0),
                    # b_k = Kicker("one turn", 250),
                    quad_model = "linear", # "linear" "full"
                    # quad_num = 1, # 1 or 4
                    quad_num = 4,
                    # n = 0,
                    n = 0.108,
                    # collimators = "continuous",
                    collimators = "discrete",
                    # collimators = "none",
                    fringe = False # turn fringe effect on or off
                )

                generator = StateGenerator(
                    mode = "mc",
                    seed = None,
                    muons = 1E7,
                    initial_offset = dist.Gaussian(mean = aux.r_inj_offset - 2.6, std = xstd),
                    # initial_offset = dist.Gaussian(mean = aux.r_inj_offset - 2.6, std = np.linspace(2, 6.4, 12)),
                    # initial_offset = dist.Gaussian(mean = aux.r_inj_offset - np.linspace(1.1, 4.1, 11), std = 4.2),
                    # initial_offset = aux.r_inj_offset,
                    # momentum = dist.Single([aux.p_magic, aux.p_magic * 0.0001]),
                    # momentum = dist.Single([aux.p_magic * 1.1, aux.p_magic]),
                    momentum = dist.Gaussian(mean = pmean, std = 0.8),
                    # momentum = dist.Gaussian(mean = 100, std = np.linspace(0.4, 1.2, 9)),
                    # momentum = dist.Gaussian(mean = np.linspace(99.5, 100.5, 11), std = 0.8),
                    # momentum = aux.p_magic * 1.003888,
                    # offset = 34,
                    # alpha = dist.Single(value = [0, 2.5]),
                    # alpha = 2.5,
                    # alpha = 0,
                    alpha = dist.Gaussian(mean = -0.9, std = alphastd),
                    # alpha = dist.Gaussian(mean = -0.9, std = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]),
                    # alpha = dist.Gaussian(mean = -0.9, std = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]),
                    # alpha = dist.Gaussian(mean = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6], std = 3.1),
                    phi_0 = 0,
                    # t = 80
                    # t = dist.Gaussian(mean = 0, std = 25)
                    t = dist.Custom(dir = "pulseshape_1_12543_mod", zero = tnorm)
                    # t = dist.Custom(dir = "pulseshape_1_12543_mod", zero = 24827.5)
                    # t = dist.Custom(dir = "pulseshape_1_12543_mod", zero = np.linspace(24805, 24850, 13))
                )

                propagator.propagate(
                    ring = ring,
                    state_generator = generator,
                    integration_method = 'rk4',
                    dt = 0.1, # the time-step in nanoseconds
                    t_f = 4000 # end time in nanoseconds
                )

                shutil.move(f"{results_path}/s2-{dir}_{xstd_idx+1}_{alphastd_idx+1}_{pmean_idx + 1}_{tnorm_idx+1}", f"{results_path}/{dir}/s2-{dir}_{xstd_idx+1}_{alphastd_idx+1}_{pmean_idx + 1}_{tnorm_idx+1}")