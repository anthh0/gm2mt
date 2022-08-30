import numba as nb
import time
import pathlib
import numba as nb
import numpy as np
import auxiliary as aux
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

split = 200
dtheta = 2 * np.pi / split
delta_p = 0.001
delta_p = 0

# constant_matrix = np.array([
#     aux.r_magic * delta_p * (1 - np.cos(dtheta)), 
#     aux.r_magic * delta_p * np.sin(dtheta),
#     dtheta,
#     ((1 + delta_p) * dtheta - delta_p * np.sin(dtheta)) / aux.omega_magic
# ])

# transfer_matrix = np.array([
#     [np.cos(dtheta), np.sin(dtheta), 0, 0],
#     [-np.sin(dtheta), np.cos(dtheta), 0, 0],
#     [0, 0, 1, 0],
#     [np.sin(dtheta) / v0, (1 - np.cos(dtheta)) / v0, 0, 1]
# ])
# init = np.array([0.020, 0, 0, 0])

constant_matrix = np.array([
    aux.r_magic * delta_p * (1 - np.cos(dtheta)), 
    aux.r_magic * delta_p * np.sin(dtheta),
    ((1 + delta_p) * dtheta - delta_p * np.sin(dtheta)) / aux.omega_magic
])

transfer_matrix = np.array([
    [np.cos(dtheta), np.sin(dtheta), 0],
    [-np.sin(dtheta), np.cos(dtheta), 0],
    [np.sin(dtheta) / aux.v_magic, (1 - np.cos(dtheta)) / aux.v_magic, 1]
])
init = np.array([0.020, 0, 0])
# steps = int(1000 / (150 / split))
steps = 1000
muons = int(1E5)

I_all = np.zeros(shape = (muons, len(init), steps+1), dtype = np.float64)
I_all[:,:,0] = init

@nb.njit(nb.float64[:, :, :](nb.float64[:,:,:], nb.float64[:,:], nb.float64[:]), fastmath = True, cache = True, parallel = True)
def nb_prop_all_states(I_all, T, C):
    m, n = T.shape
    muons, _, steps = I_all.shape
    for k in nb.prange(muons):
        for t in nb.prange(steps-1):
            for i in nb.prange(m):
                for j in nb.prange(n):
                    I_all[k, i, t+1] += T[i,j]*I_all[k, j, t] + (C[i] / m)
    return I_all

print("Starting...")
begin_time = time.time()
stepped = nb_prop_all_states(I_all, transfer_matrix, constant_matrix)
print(f"Numba (storing all) took {time.time()-begin_time}")

print(stepped[1,:,-1])


##########################################################################################################

muons = int(1E6)
I_all_new = np.zeros(shape = (muons, len(init)), dtype = np.float64)
I_all_new[:,:] = init


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:, :], nb.float64[:], nb.int64), nogil = True, fastmath = True, cache = True, parallel = False)
def forward(I_0, T, C, steps):
    m = I_0.shape[0]
    I_prime = np.zeros(shape = m)
    for t in range(steps):
        # for i in nb.prange(m):
        for i in range(m):
            # for j in nb.prange(n):
            for j in range(m):
                I_prime[i] += ((T[i,j]*I_0[j]) + (C[i] / m))
        I_0[:] = I_prime[:]
        I_prime[:] = 0
    return I_0[:]

@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.float64[:]), nogil = True, fastmath = True, cache = True, parallel = False)
def nb_prop(I, T, C):
    steps = 1000
    muons, m = I.shape
    I_0 = np.zeros(shape = m)
    for k in nb.prange(muons):
    # for k in range(muons):
        # print(k)
        I_0[:] = I[k,:]
        I[k,:] = forward(I_0, T, C, steps)
    return I



print("Starting...")
begin_time = time.time()
stepped2 = nb_prop(I_all_new, transfer_matrix, constant_matrix)
print(f"Numba (store final) took {time.time()-begin_time}")

print(stepped2[0,:])
print(stepped2[1,:])
print(stepped2[2,:])
print(stepped2[3,:])
print(stepped2[4,:])
print(stepped2[5,:])
print(stepped2[30,:])

# x = stepped[0,0,:] * 1000
# x_prime = stepped[0,1,:]
# theta = stepped[0,2,:]
# t = stepped[0,3,:] * 10**9

# fig, ax = plt.subplots()
# ax.plot(t, x)
# # plt.show()

# directory = str(pathlib.Path(__file__).parent.resolve())
# plot = PdfPages(directory+"/plots_transfer.pdf")
# plot.savefig(fig)
# plot.close()


# nb_dot.parallel_diagnostics(level = 1)