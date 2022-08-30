from tokenize import _all_string_prefixes
import numpy as np
import pathlib

results_path = str(pathlib.Path(__file__).parent.absolute()) + "/injection_profiles"
total_centers = 0
total_heights = 0
for i in range(8):
    bin_centers, bin_heights = np.loadtxt(fname = f"{results_path}/pulseshape_{i}_12543_mod.dat", skiprows = 5, usecols = (0, 1), unpack = True)
    total_heights = total_heights + bin_heights

avg_heights = total_heights / 8
bin_centers = bin_centers.astype(int)
# print(len(avg_heights))
np.savetxt(f"{results_path}/pulseshape_avg_12543_mod.dat", np.array([bin_centers, avg_heights]).T, delimiter=' ', fmt = ['%5.0i', '.2f'])

# # Curve 0 of 1, 601 points
# # Curve title: "'pulseshape_0_12543.dat' u ($1-6220):($2+1860)*5"
# # x y type
#  NaN  NaN  u