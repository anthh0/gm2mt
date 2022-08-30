import numba as nb
import time
import pathlib
import numba as nb
import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights
import auxiliary as aux
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



dtheta = 2 * np.pi / 200
# delta_p = 0.001
delta_p = 0
delta_x = 0.020 / aux.r_magic
delta_omega = -delta_x
constant_matrix = np.array([aux.r_magic * delta_p * (1 - np.cos(dtheta)), aux.r_magic * delta_p * np.sin(dtheta), ((1 + delta_p) * dtheta - delta_p * np.sin(dtheta)) / aux.omega_magic])
v0 = aux.v_magic
transfer_matrix = np.array([
    [np.cos(dtheta), np.sin(dtheta), 0],
    [-np.sin(dtheta), np.cos(dtheta), 0],
    [-np.sin(dtheta) / v0, (1 - np.cos(dtheta)) / v0, 1]
])
init = np.array([0.020, 0, 0])

steps = 1000
muons = int(1E5)
all_inits = np.tile(init, (muons, 1))

@nb.njit(nb.float64[:, :](nb.float64[:,:], nb.float64[:,:], nb.float64[:]), nogil = True, fastmath = True, cache = True, parallel = True)
def numpydot(T, inits, C):
    I = np.zeros(3)
    for j in nb.prange(muons):
        I[:] = inits[j,:]
        for i in range(steps):
            I = T @ I + C
        inits[j,:] = I[:]
    return inits

print("Starting...")
begin_time = time.time()
stepped = numpydot(transfer_matrix, all_inits, constant_matrix)
end_time = time.time()

print(f"Numpy dot took {end_time-begin_time}")

print(stepped[0,:])
# print(aux.omega_magic*stepped[0,1]*1E-6)
# x = stepped[0,:] * 1000
# x_prime = stepped[0,1,:]
# t = stepped[0,2,:] * 10**9

# fig, ax = plt.subplots()
# ax.plot(t, x)
# # plt.show()

# directory = str(pathlib.Path(__file__).parent.resolve())
# plot = PdfPages(directory+"/plots_transfer.pdf")
# plot.savefig(fig)
# plot.close()