import scipy.optimize as opt
import numpy as np
from gm2mt.StateGenerator import StateGenerator
from gm2mt.Plotter import Plotter
import shutil
import gm2fr.constants as const
import pathlib
import gm2mt.auxiliary as aux

dir = "fullscanLarge"
full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/"

xstds = np.linspace(3, 5.4, 5)
alphastds = np.linspace(2.4, 3.8, 5)
pmeans = np.linspace(99.7, 100.3, 5)
tnorms = np.linspace(24805, 24850, 5)

# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             for l in range(5):
#                 shutil.move(f"{full_path}s2-{dir}_{i+1}_{j+1}_{k+1}_{l+1}", f"{full_path}{dir}/s2-{dir}_{i+1}_{j+1}_{k+1}_{l+1}")

# np.savez(f"{full_path}{dir}/params.npz", pmean = pmeans, tnorm = tnorms, alphastd = alphastds, xstd = xstds)
with open(f"{full_path}fullscan/directory.txt", "w") as f:
    # f.write(f"0 xstd\n1 alphastd\n2 pmean\n3 tnorm")
    f.write(f"0 pmean\n1 tnorm")



# results = np.load("results/s2-alg_source_1e7_100k/final_states.npz")
# lost = results['lost']
# r = results['r']
# vr = results['vr']
# vphi = results['vphi']

# width = 150
# df = 2
# n=0.108

# edges = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)

# f = aux.p_to_f(aux.state_to_mom_cyl(r = r, vr = vr, vphi = vphi), n = n)
# f_s = Plotter._mask_lost(lost, f)
# h, edges = np.histogram(f_s, bins = edges)
# errors = np.sqrt(h) / (len(f_s) * df)
# h_normed = h / (len(f_s) * df)
# assert sum(h) == len(f_s)
# cov = np.diag(np.square(errors))
# np.savez(f"results/s2-alg_source_1e7_100k/f_dist.npz", heights = h_normed, cov = cov)