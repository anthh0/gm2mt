import pathlib
import numpy as np
import matplotlib.pyplot as plt
import time
import auxiliary as aux

#region: X-A CORRELATION
# fig, ax = plt.subplots()

# w = 1
# l = 10
# num_points = 1000

# xlist = np.linspace(-w/2, w/2, num_points)
# dx = w / num_points

# def aL(x):
#     return -np.arctan((w/2-x)/l)

# def aU(x):
#     return np.arctan((w/2+x)/l)

# a_max = aU(w/2)
# a_min = aL(-w/2)
# alpha = np.linspace(a_min-0.1, a_max+0.1, num_points)

# intensity = np.zeros(num_points)

# # Defines the intensity distribution as a function of angle and offset
# def p_a(alpha, x):
#     if aL(x) < alpha and alpha < aU(x):
#         return 1/(aU(x) - aL(x))
#     else:
#         return 0.0

# # Defines the x distribution weights
# def p_x(x):
#     # if -w/2 <= x and x <= 0:                # triangle dist
#     #     return 4*x/w**2 + 2/w
#     # elif 0 < x and x <= w/2:
#     #     return -4*x/w**2 + 2/w
#     # else:
#     #     return 0.0
#     # if -w/2 <= x and x <= 0:                # inverted triangle dist
#     #     return -4*x/w**2
#     # elif 0 < x and x <= w/2:
#     #     return 4*x/w**2
#     # else:
#     #     return 0.0
#     # if -w/2 <= x and x <= w/2:              # constant dist
#     #     return 1/w
#     # else:
#     #     return 0.0
#     if -w/2 <= x and x <= w/2:              # gaussian dist
#         return np.exp(-x**2/(w/5)**2/2)
#     else:
#         return 0.0

# # Integrate the contribution to intensity over x as a function of angle
# for x in xlist:
#     for i in range(len(alpha)):
#         a = alpha[i]
#         intensity[i] += p_a(a, x)*p_x(x)*dx

# # Integrate the x-weights for normalization
# X = 0
# for x in xlist:
#     X += p_x(x)*dx

# intensity /= X

# ax.plot(alpha, intensity, 'b')
# ax.grid()
# ax.set_title("Intensity vs Angle")
# ax.set_xlabel("Angle")
# ax.set_ylabel("intensity")

# plt.show()
#endregion

#region: ITERATIVE LAPLACE SOLVER
# W = 200                                     # dimensions of bounding region
# H = 120

# s = 100                                     # dimensions of ESQ plates
# w = 47

# V = 1                                       # voltage on plates

# dx = 1
# dy = 1
# Nx = int(W / dx)
# Ny = int(H / dy)

# potential = np.zeros(shape = (Ny, Nx))

# def applyBCs(pot_field):
#     result = pot_field.copy()

#     for y in range(Ny):
#         for x in range(Nx):
#             result[y][0] = 0
#             result[0][x] = 0
#             result[y][-1] = 0
#             result[-1][x] = 0
    
#     for y_idx in range(int(Ny/H * (H/2 - w/2)), int(Ny/H * (H/2 + w/2))):
#         x_l_idx = int(Nx/W * (W/2 - s/2))
#         x_u_idx = int(Nx/W * (W/2 + s/2))

#         result[y_idx][x_l_idx] = -V
#         result[y_idx][x_u_idx] = -V
    
#     for x_idx in range(int(Nx/W * (W/2 - w/2)), int(Nx/W * (W/2 + w/2))):
#         y_l_idx = int(Ny/H * (H/2 - s/2))
#         y_u_idx = int(Ny/H * (H/2 + s/2))

#         result[y_l_idx][x_idx] = V
#         result[y_u_idx][x_idx] = V
    
#     return result

# def update(pot_field):
#     temp_field = np.zeros(shape = (Ny, Nx))
#     for y_idx in range(1, Ny-1):
#         for x_idx in range(1, Nx-1):
#             val1 = pot_field[y_idx-1][x_idx]
#             val2 = pot_field[y_idx+1][x_idx]
#             val3 = pot_field[y_idx][x_idx-1]
#             val4 = pot_field[y_idx][x_idx+1]
#             temp_field[y_idx][x_idx] = (val1 + val2 + val3 + val4)/4
#     temp_field = applyBCs(temp_field)
#     return temp_field

# def get_Er(pot_field):
#     neg_grad = np.zeros(shape = Nx)
#     mid_y_idx = int(Ny/2)

#     neg_grad[0] = -(pot_field[mid_y_idx][1] - pot_field[mid_y_idx][0])/dx
#     for x_idx in range(1, Nx-1):
#         neg_grad[x_idx] = -(pot_field[mid_y_idx][x_idx+1] - pot_field[mid_y_idx][x_idx-1])/(2*dx)
#     neg_grad[-1] = -(pot_field[mid_y_idx][-1] - pot_field[mid_y_idx][-2])/dx

#     mid_x_idx = int(Nx/2)
#     m = (neg_grad[mid_x_idx+1] - neg_grad[mid_x_idx-1])/(2*dx)

#     return neg_grad, m


# potential = applyBCs(potential)
# for n in range(1000):
#     potential = update(potential)

# Er, m = get_Er(potential)
# xc = np.linspace(-W/2, W/2, Nx)
# yc = m*xc

# fig, axs = plt.subplots(1, 2)

# axs[0].imshow(potential, cmap = 'bwr', interpolation = None)
# axs[1].plot(xc, Er, 'b.')
# axs[1].plot(xc, yc, 'k')
# axs[1].grid()
# plt.show()
#endregion

#region: Integrator e_eval comparison
# def e_eval_linear_1_interior(angle, r, k_e):
#     if r >= 7.062 and r <= 7.162:
#         return (k_e * (r - 7.112))
#     else:
#         e_str = 0.0
#         return e_str

# def e_eval_full_1_interior(angle, r, k_e):
#     s = 0.1
#     w = 0.047
#     sig = k_e*(s**2 + w**2)/32/w
#     if r >= 7.062 and r <= 7.162:
#         r_o = r - 7.112
#         return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
#     else:
#         return 0.0

# rc = np.linspace(7.062, 7.162, 100)
# k_e = 1
# e_linear = [e_eval_linear_1_interior(angle = 0, r = r, k_e = k_e) for r in rc]
# e_full = [e_eval_full_1_interior(angle = 0, r = r, k_e = k_e) for r in rc]

# fig, ax = plt.subplots()
# ax.plot(rc, e_linear, 'b', label = 'Linear')
# ax.plot(rc, e_full, 'r', label = 'Full')

# ax.set_title("ESQ Field Model Comparison")
# ax.grid()
# ax.set_xlim([7.062, 7.162])
# ax.set_xlabel("Radius [m]")
# ax.set_ylabel(r"E_r [V/m]")
# ax.legend()

# plt.show()
#endregion

#region: X-A correlation plotting
source = str(pathlib.Path(__file__).parent.absolute()) + "/x_alpha_profiles/PostInflectorFull_RubinClean.dat"

bins = 100
hrange = 10

x, px, pz = np.loadtxt(fname = source, skiprows = 1, usecols = (3, 4, 8), unpack = True)
x = 1E3*x - aux.r_inj_offset        # convert (ring) offset from [m] to (inflector) offset [mm]
Px = px*aux.p_magic                 # convert from phase space momentum to real momentum
Pz = pz*aux.p_magic + aux.p_magic
alpha = 1E3*np.arctan(Px / Pz)      # calculate inflector angle [mrad]

print(f"mean x: {np.mean(x)}, std x: {np.std(x)}")
print(f"mean a: {np.mean(alpha)}, std a: {np.std(alpha)}")

x_hist1d_normed, x_edges_1 = np.histogram(x, bins = bins, range = [-hrange, hrange], density = True)
x_alpha_hist2d, x_edges_2, a_edges_2 = np.histogram2d(x, alpha, bins = bins, range = [[-hrange, hrange], [-hrange, hrange]])

a_dists = []
for i in range(bins):
    a_hist1d_normed = x_alpha_hist2d[i] / x_alpha_hist2d[i].sum()
    a_dists.append(x_alpha_hist2d[i])

# fig, axs = plt.subplots(1, 2)
# axs[0].plot(np.linspace(-hrange, hrange, bins), x_hist1d_normed)
# for idx, a_dist in enumerate(a_dists):
#     if idx % 10 == 0:
#         axs[1].plot(np.linspace(-hrange, hrange, bins), a_dist)
fig, ax = plt.subplots()
ax.imshow(x_alpha_hist2d.T, cmap = 'plasma')
ax.plot(np.full(shape = bins, fill_value = (np.mean(x) + hrange)/(2*hrange)*bins), np.linspace(0, bins-1, bins), 'b', label = r"Mean $x$")
ax.plot(np.linspace(0, bins-1, bins), np.full(shape = bins, fill_value = (np.mean(alpha) + hrange)/(2*hrange)*bins), 'r', label = r"Mean $\alpha$")
ax.legend()
ax.set_title("Initial Offset vs Angle")
ax.set_xlabel("Initial Offset [mm]")
ax.set_ylabel("Initial Angle [mrad]")
# ax.set_xlim([-hrange, hrange])
# ax.set_ylim([-hrange, hrange])
plt.show()
