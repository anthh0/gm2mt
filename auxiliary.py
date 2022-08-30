# from selectors import EpollSelector
import numpy as np

#region CONSTANTS:
m = 1.883531627E-28 # muon mass (kg)
q = 1.602176634E-19 # muon charge (C)
c = 299792458 # speed of light (m/s)
B_nom = 1.4513 # nom. field strength (T)
gev_to_kgms = 5.344286E-19 # Multiplicative factor for converting FROM GeV to kg*m/s
#endregion

#region MAGIC PARAMETERS:
r_magic = 7.112 # Magic radius (m)
r_inj = 7.189 # Injector radial position (m)
r_inj_offset = 77 # Injector radial offset (mm)
p_magic = r_magic * q * B_nom / gev_to_kgms # Magic momentum (GeV)
_ = (p_magic * gev_to_kgms / m / c)**2 # This is just a helper variable for the magic velocity calculation
v_magic = np.sqrt(_ / (1 + _)) * c # Magic velocity (m/s)
f_magic = q * B_nom * v_magic / (2 * np.pi * p_magic * gev_to_kgms) / 1000 # Magic frequency (kHz)
T_magic = 2 * np.pi * p_magic * gev_to_kgms / q / B_nom / v_magic # Magic period (s)
omega_magic = q * B_nom * v_magic / (p_magic * gev_to_kgms) # Magic angular speed (rad/s)
gamma_magic = 1 / np.sqrt(1 - v_magic*v_magic / c / c) # Magic boost factor (unitless)
beta_magic = v_magic / c
#endregion

#region KICKER LOCATIONS:
# All kicker locations below are given in radians.
k1_i = 1.1855314960629921 # 67.92595120423982 = 8.4315 / 7.112 * 180 / np.pi
k1_f = 1.3641029246344205 # 78.15734040300451 = 9.7015 / 7.112 * 180 / np.pi
k2_i = 1.4471316085489314 # 82.91453356983173 = 10.292 / 7.112 * 180 / np.pi
k2_f = 1.6257030371203598 # 93.14592276859642 = 11.562 / 7.112 * 180 / np.pi
k3_i = 1.7086614173228345 # 97.89908782943986 = 12.152 / 7.112 * 180 / np.pi
k3_f = 1.8872328458942633 # 108.13047702820458 = 13.422 / 7.112 * 180 / np.pi

k3_f_no_gaps = 1.72124578178 # kicker end placement with kicker gaps removed
#endregion

#region QUADRUPOLE LOCATIONS:
# All quadrupole locations below are given in radians.
Q1S_i = 0.3852784026996625 # 2.7401 / 7.112 
Q1S_f = 0.6121766029246344 # 4.3538 / 7.112 
Q1L_i = 0.6819881889763779 # 4.8503 / 7.112 
Q1L_f = 1.1357705286839146 # 8.0776 / 7.112 

Q2S_i = 1.9561304836895388 # 13.912 / 7.112 
Q2S_f = 2.1829302587176604 # 15.525 / 7.112
Q2L_i = 2.2528121484814396 # 16.022 / 7.112
Q2L_f = 2.7065523059617544 # 19.249 / 7.112

Q3S_i = 3.52685601799775   # 25.083 / 7.112
Q3S_f = 3.753796400449944  # 26.697 / 7.112
Q3L_i = 3.8235376827896514 # 27.193 / 7.112
Q3L_f = 4.277418447694038  # 30.421 / 7.112

Q4S_i = 5.097722159730034  # 36.255 / 7.112 
Q4S_f = 5.324521934758155  # 37.868 / 7.112
Q4L_i = 5.394403824521935  # 38.365 / 7.112
Q4L_f = 5.84814398200225   # 41.592 / 7.112

quad_lims = {'Q1': (Q1S_i, Q1L_f), 'Q2': (Q2S_i, Q2L_f), 'Q3': (Q3S_i, Q3L_f), 'Q4': (Q4S_i, Q4L_f)}
plate_len = 0.047
plate_sep = 0.100
#endregion

#region ADDITIONAL QUADRUPOLE VALUES:
tot_quad_length = 19.363 # Total length of ESQ plates (m) 
    # Calculation: 4.3538 - 2.7401 + 8.0776 - 4.8503 + 15.525 - 13.912 + 19.249 - 16.022 + 26.697 - 25.083 + 30.421 - 27.193 + 37.868 - 36.255 + 41.592 - 38.365
quad_coverage = 0.4333123120202994 # Total angular coverage of ESQ plates (radians)
    # Calculation: tot_quad_length / (2 * np.pi * r_magic)
#endregion

#region COLLIMATOR LOCATIONS:
# All positions given in radians.
C1 = 2.217941507311586 # 15.774 / 7.112
C2 = 2.7065523059617544 # 19.249 / 7.112
C3 = 4.277418447694038 # 30.421 / 7.112
C4 = 5.359533183352081 # 38.117 / 7.112
C5 = 5.84814398200225 # 41.592 / 7.112
#endregion

#------------------------------------------------------------------------------------------------------------------#

# FUNCTIONS: Conversion functions for various variables.  Check the docstrings for details like units.

def p_to_v(momentum):
    """Convert from momentum (GeV) to velocity (m/s)."""
    A = (momentum * gev_to_kgms / m / c)**2
    v = np.sqrt(A / (1 + A)) * c
    return v

def v_to_p(v):
    """Convert from velocity (m/s) to momentum (GeV)"""
    return m * v / np.sqrt(1 - (v / c)**2) / gev_to_kgms

def p_to_f(momentum, n):
    """Convert from momentum (GeV) to frequency (kHz)."""
    return f_magic * (1 - 1 / (1 - n) * (momentum - p_magic) / p_magic)

def f_to_v(f, n):
    """Convert from frequency (kHz) to velocity (m/s)."""
    p = p_magic * (1 + (1 - n) * (1 - f / f_magic))
    return p_to_v(p)

def p_to_rco(p, n):
    """Convert from momentum (GeV) to radial offset (mm)."""
    return r_magic / (1 - n) * (p - p_magic) / p_magic * 1E3

def rco_to_v(offset, n):
    """Convert from equilibrium radius (mm) to velocity (m/s)."""
    p = p_magic * (offset * (1-n) / r_magic / 1000 + 1)
    return p_to_v(p)

def state_to_mom_cyl(r, vr, vphi):
    """Convert from a given state in cylindrical coordinates to its momentum (GeV)."""
    v_sq = (vr * vr) + (r * r * vphi* vphi)
    gamma = 1 / np.sqrt(1 - (v_sq / c / c ))
    return gamma * m * np.sqrt(v_sq) / gev_to_kgms

def state_to_mom_cart(vx, vy):
    """Convert from a given state in Cartesian coordinates to its momentum (GeV)."""
    v = np.sqrt(vx**2 + vy**2)
    gamma = 1 / np.sqrt(1 - ((vx**2 + vy**2) / (c**2)))
    return gamma * m * v / gev_to_kgms

def mask_zeroes(array, *related_arrays):
    """Mask all zeroes at the front and back of an array, and mask the same range on related arrays."""
    idxs = array.nonzero()[0]
    front, back = idxs[0], idxs[-1]
    masked_array = array[front:back + 1]
    related_arrays_list = list(related_arrays)
    masked_related_arrays_list = [i[front:back+1] for i in related_arrays_list]
    full_array_list = [masked_array] + masked_related_arrays_list
    return tuple(full_array_list)

def x_to_C_E(avg_x, sigma_x, n):
    """Convert mean and width of radial distribution (mm) into electric field correction (ppb)"""
    return 2 * n * (1 - n) * beta_magic**2 / r_magic**2 * (avg_x**2 + sigma_x**2) * 10**3

def n_to_voltage_nonvectorized(n, quad_num, r = 0.050):
    """Convert field index to ESQ plate voltage (V), taking into account continuous or discrete quads."""
    if quad_num == 1:
        return n * v_magic * B_nom * r * r / (2 * r_magic)
    elif quad_num == 4:
        return n * v_magic * B_nom * r * r / (2 * r_magic * quad_coverage)

def n_to_voltage(n, quad_num):
    n_to_voltage_2 = np.vectorize(n_to_voltage_nonvectorized, otypes = [float])
    voltage = n_to_voltage_2(n, quad_num)
    if isinstance(voltage, np.ndarray) and voltage.shape == ():
        return voltage.item()
    else:
        return voltage

def n_to_k_nonvectorized(n, quad_num):
    """Convert from field index to electric field spring constant (V/m^2), taking into account continuous or discrete quads."""
    if quad_num == 1:
        k = n * v_magic * B_nom / r_magic
    if quad_num == 4:
        k = n * v_magic * B_nom / r_magic / quad_coverage # I'm not sure why this would be but Tucker wrote it so i'm not going to change it (yet) #
    return k

def n_to_k(n, quad_num):
    n_to_k_2 = np.vectorize(n_to_k_nonvectorized, otypes = [float])
    k = n_to_k_2(n, quad_num)
    if isinstance(k, np.ndarray) and k.shape == ():
        return k.item()
    else:
        return k

def delist(obj):
    if isinstance(obj, (list, np.ndarray)):
        if len(obj) == 1:
            return obj[0]
        else:
            raise IndexError(f"Attempted delisting; list/NumPy array is of length {len(obj)}")
    else:
        return obj