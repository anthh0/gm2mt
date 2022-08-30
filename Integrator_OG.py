import numpy as np
import numba as nb
import math
import gm2mt.auxiliary as aux

m = 1.883531627E-28 # muon mass in kg
q = 1.602176634E-19 # muon charge in C
c = 299_792_458 # speed of light in m/s

#region B AND E FIELD FUNCTIONS:
@nb.njit(nb.boolean(nb.float64), fastmath = True, cache = True)
def inQuad(angle): # ring angle (NOT Cartesian angle)
    angle = angle % (2 * np.pi)
    if (angle > 0.3852784026996625 and angle < 1.1357705286839146) or (angle > 1.9561304836895388 and angle < 2.7065523059617544) or (angle > 3.52685601799775 and angle < 4.277418447694038) or (angle > 5.097722159730034 and angle < 5.84814398200225):
    # if (angle > 0.3852784026996625 and angle < 0.6121766029246344) or (angle > 0.6819881889763779 and angle < 1.1357705286839146) or (angle > 1.9561304836895388 and angle < 2.1829302587176604) or (angle > 2.2528121484814396 and angle < 2.7065523059617544) or (angle > 3.52685601799775 and angle < 3.753796400449944) or (angle > 3.8235376827896514 and angle < 4.277418447694038) or (angle > 5.097722159730034 and angle < 5.324521934758155) or (angle > 5.394403824521935 and angle < 5.84814398200225):
        return True
    else:
        return False
    
@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_1(angle):
    angle = angle % (2 * np.pi)
    if angle > 1.1855314960629921 and angle < 1.72124578178:
        return 1.1855314960629921, 1.72124578178
    else:
        return 0.0, 0.0

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_3(angle):
    angle = angle % (2 * np.pi)
    if angle > 1.1855314960629921 and angle < 1.3641029246344205:
        return 1.1855314960629921, 1.3641029246344205
    elif angle > 1.4471316085489314 and angle < 1.6257030371203598:
        return 1.4471316085489314, 1.6257030371203598
    elif angle > 1.7086614173228345 and angle < 1.8872328458942633:
        return 1.7086614173228345, 1.8872328458942633
    else:
        return 0.0, 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_linear_1(angle, r, k_e):
    if r > 7.062 and r < 7.162:
        return (k_e * (r - 7.112))
    else:
        e_str = 0.0
        return e_str

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_linear_4(angle, r, k_e):
    if inQuad(angle) and r > 7.062 and r < 7.162:
        return (k_e * (r - 7.112))
    else:
        e_str = 0.0
        return e_str

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_full_interior_1(angle, r, k_e):
    s = 0.1
    w = 0.047
    sig = k_e*(s**2 + w**2)/32/w
    if r > 7.062 and r < 7.162:
        r_o = r - 7.112                     # r offset from r_magic [m]
        return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
    else:
        return 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_full_exterior_1(angle, r, k_e):
    s = 0.1
    w = 0.047
    sig = k_e*(s**2 + w**2)/32/w
    r_o = r - 7.112                         # r offset from r_magic [m]
    return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_full_interior_4(angle, r, k_e):
    s = 0.1
    w = 0.047
    sig = k_e*(s**2 + w**2)/32/w
    if inQuad(angle) and r > 7.062 and r < 7.162:
        r_o = r - 7.112                     # r offset from r_magic [m]
        return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
    else:
        return 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_full_exterior_4(angle, r, k_e):
    s = 0.1
    w = 0.047
    sig = k_e*(s**2 + w**2)/32/w
    if inQuad(angle):
        r_o = r - 7.112                     # r offset from r_magic [m]
        return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
    else:
        return 0.0

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def cyl_deriv(r, vr, vphi, B, E):
    m = 1.883531627E-28 # muon mass in kg
    q = 1.602176634E-19 # muon charge in C
    c = 299_792_458 # speed of light in m/s
    gamma = 1 / np.sqrt(1 - (( (vr*vr) + (r*r*vphi*vphi) )  / c / c))
    ar = (r*vphi*vphi) + ((q / gamma / m) * (E - (B * r * vphi) - vr * vr * E / c / c  ) )
    aphi = ((q / gamma / m * (B * vr - vr * E * r * vphi / c / c)) - (2 * vr * vphi)) / r
    return ar, aphi
#endregion

#region EXIT CHECKER FUNCTIONS:
@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_cont(r, phi, t_step, r_max, r_min):
    # if (r > r_max or r < r_min) and inKicker_3(phi) == (0.0, 0.0):
    if (r > r_max or r < r_min):
        return False
    else:
        return True

@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_disc(r, phi, t_step, r_max, r_min):
    phi = phi % (2 * np.pi)
    width = (2 * np.pi / (1.5E-7 / t_step))
    C1 = 2.217941507311586
    C2 = 2.7065523059617544
    C3 = 4.277418447694038
    C4 = 5.359533183352081
    C5 = 5.84814398200225
    if (abs(phi - C1) < width or abs(phi - C2) < width or abs(phi - C3) < width or abs(phi - C4) < width or abs(phi - C5) < width) and (r > r_max or r < r_min):
        return False
    else:
        return True

@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_none(r, phi, t_step, r_max, r_min):
    return True
#endregion

#region RK4 KICKER FUNCTIONS:
@nb.njit(fastmath = True, cache = True)
def tstep_init(r, phi, vr, vphi, b_nom, target, k_e, e_func, original_t_step):
    jump_width = original_t_step / 10
    t_step = jump_width
    while t_step <= original_t_step:
        E1 = e_func(phi, r, k_e)
        vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
        E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
        vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
        E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
        vphi3, aphi3 = vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)[1]

        vphi4 = vphi + t_step * aphi3

        final_angle = phi + t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
        if final_angle > target:
            break
        else:
            t_step += jump_width
    return t_step

@nb.njit(fastmath = True, cache = True)
def kicker_s(states, j, r_max, r_min, b_nom, b_k, t_list, k_i, k_f, k_e, e_func, t_step, t_f, exit_checker, inj_region):
    # EDGE MATCHING, first iteration - match the timestep to the leading kicker edge
    r = states[j][0]
    phi = states[j][1]
    vr = states[j][2]
    vphi = states[j][3]
    t = states[j][4]
    
    original_t_step = t_step
    t_step = tstep_init(r, phi % (2 * np.pi), vr, vphi, b_nom, k_i, k_e, e_func, original_t_step)
    E1 = e_func(phi, r, k_e)
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step

    # EDGE MATCHING, second iteration - fill out the rest of the timestep

    t_step = original_t_step - t_step
    
    B1, B2, B3, B4 = b_nom - np.interp(t, t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + t_step, t_list, b_k)

    E1 = e_func(phi, r, k_e)
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step
    j += 1

    states[j][0] = r
    states[j][1] = phi
    states[j][2] = vr
    states[j][3] = vphi
    states[j][4] = t

    # KICKER PATCHING: finely propagate through the kicker region, with "split" defining the fineness;
    #                  loop ends when it reaches the sim lifetime end or exits the kicker
    split = 5
    t_list_fine = np.arange(t, t + 2E-8, original_t_step / (2 * split) )
    b_k_fine = np.interp(t_list_fine, t_list, b_k)
    t_step = original_t_step / split
    kicker_idx = 0
    lost_in_kicker = False

    while t < t_f and (phi % (2 * np.pi)) < k_f:
        if phi > inj_region and not exit_checker(r, phi, original_t_step, r_max, r_min):
            lost_in_kicker = True
            break

        states[j][0] = r
        states[j][1] = phi
        states[j][2] = vr
        states[j][3] = vphi
        states[j][4] = t

        for i in nb.prange(split):
            B1 = b_nom - b_k_fine[2 * kicker_idx]
            E1 = e_func(phi, r, k_e)
            vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)

            B2 = b_nom - b_k_fine[2 * kicker_idx + 1]
            E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
            vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)

            B3 = b_nom - b_k_fine[2 * kicker_idx + 1]
            E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
            vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)

            B4 = b_nom - b_k_fine[2 * kicker_idx + 2]
            E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
            vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

            r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
            phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
            vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
            vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
            t += t_step
            kicker_idx += 1

            if (phi % (2 * np.pi)) > k_f:
                t_step = (split - i) * t_step

                E1 = e_func(phi, r, k_e)
                vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)

                E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
                vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)

                E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
                vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)

                E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
                vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

                r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
                phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
                vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
                vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
                t += t_step

                break
        
        j += 1
    
    return r, phi, vr, vphi, t, lost_in_kicker, j

@nb.njit(fastmath = True, cache = True)
def kicker_m(r, phi, vr, vphi, t, r_max, r_min, b_nom, b_k, t_list, k_i, k_f, k_e, e_func, t_step, t_f, exit_checker, inj_region):
    # EDGE MATCHING, first iteration - match the timestep to the leading kicker edge
    
    original_t_step = t_step
    t_step = tstep_init(r, phi % (2 * np.pi), vr, vphi, b_nom, k_i, k_e, e_func, original_t_step)
    E1 = e_func(phi, r, k_e)
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step

    # EDGE MATCHING, second iteration - fill out the rest of the timestep

    t_step = original_t_step - t_step
    
    B1, B2, B3, B4 = b_nom - np.interp(t, t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + t_step, t_list, b_k)

    E1 = e_func(phi, r, k_e)
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step

    # KICKER PATCHING: finely propagate through the kicker region, with "split" defining the fineness;
    #                  loop ends when it reaches the sim lifetime end or exits the kicker
    
    split = 5
    t_list_fine = np.arange(t, t + 2E-8, original_t_step / (2 * split) )
    b_k_fine = np.interp(t_list_fine, t_list, b_k)
    t_step = original_t_step / split
    kicker_idx = 0
    lost_in_kicker = False

    while t < t_f and (phi % (2 * np.pi)) < k_f:
        if phi > inj_region and not exit_checker(r, phi, original_t_step, r_max, r_min):
            lost_in_kicker = True
            break

        B1 = b_nom - b_k_fine[2 * kicker_idx]
        E1 = e_func(phi, r, k_e)
        vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)

        B2 = b_nom - b_k_fine[2 * kicker_idx + 1]
        E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
        vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)

        B3 = b_nom - b_k_fine[2 * kicker_idx + 1]
        E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
        vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)

        B4 = b_nom - b_k_fine[2 * kicker_idx + 2]
        E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
        vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

        r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
        phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
        vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
        vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
        t += t_step
        kicker_idx += 1
    
    return r, phi, vr, vphi, t, lost_in_kicker
#endregion

class Integrator:
    def __init__(self, integration_method):
        self.integration_method = integration_method
    
    def integrate(self, init_states, ring, dt, t_f):
        dt /= 10**9
        t_f /= 10**9
        inj_region = 2.217941507311586 # location of C1 (rad)

        if ring.quad_num == 1:                                                              # select the appropiate ESQ e field evaluator
            if ring.quad_model == "linear":
                e_func = e_eval_linear_1
            elif ring.quad_model == "full_interior":
                e_func = e_eval_full_interior_1
            elif ring.quad_model == "full_exterior":
                e_func = e_eval_full_exterior_1
        elif ring.quad_num == 4:
            if ring.quad_model == "linear":
                e_func = e_eval_linear_4
            elif ring.quad_model == "full_interior":
                e_func = e_eval_full_interior_4
            elif ring.quad_model == "full_exterior":
                e_func = e_eval_full_exterior_4

        if ring.collimators == "continuous":                                                # select the appropriate exit checker
            exit_checker = exit_check_cont
        elif ring.collimators == "discrete":
            exit_checker = exit_check_disc
        else:
            exit_checker = exit_check_none
        
        if ring.b_k.kicker_num == 1:                                                        # select the appropriate kicker checker
            inKicker_func = inKicker_1
        elif ring.b_k.kicker_num == 3:
            inKicker_func = inKicker_3
        else:
            raise ValueError("kickernum unrecognized")

        init_state_c = init_states.copy() # original init_state should NEVER be touched; do any operations on this copied version
        if len(init_state_c) == 1:
            init_state_c = init_state_c[0]
            times = np.arange(init_state_c[4], t_f, dt)
            b_k = np.interp(times, (ring.b_k).t_list, (ring.b_k).b_k)
            if self.integration_method == "rk4":
                integrator_func = Integrator._jump_rk4_s
            elif self.integration_method == "optical":
                pass
            # RETURNS: states array, lost
        else:
            times = ring.b_k.t_list
            b_k = ring.b_k.b_k
            if self.integration_method == "rk4":
                integrator_func = Integrator._jump_rk4_m
            elif self.integration_method == "optical":
                pass
            # RETURNS: final_states, lost array
        
        return integrator_func(
            state = init_state_c, # a NumPy array for the "in"-state
            r_max = ring.r_max,
            r_min = ring.r_min,
            b_nom = ring.b_nom,
            b_k = b_k,
            t_list = times,
            pseudo_kick_max = ring.b_k.kick_max * ring.b_k.kicker_num,
            inKicker_func = inKicker_func,
            k_e = ring.k_e,
            e_func = e_func,
            t_f = t_f,
            t_step = dt,
            inj_region = inj_region,
            exit_checker = exit_checker)

    @staticmethod
    # @nb.njit(fastmath = True, cache = True)
    def _jump_rk4_s(state, r_max, r_min, b_nom, b_k, t_list, pseudo_kick_max, inKicker_func, k_e, e_func,
        t_f, t_step, inj_region, exit_checker):
        r = state[0]
        phi = state[1]
        vr = state[2]
        vphi = state[3]
        t = state[4]

        max_jumps = int( (t_f - t) // t_step) + 1
        states = np.empty(shape = (max_jumps, 5))
        kick_counter, j = 0, 0
        lost = False
        while j < max_jumps:
            if phi > inj_region and not exit_checker(r, phi, t_step, r_max, r_min):
                lost = True
                break
            
            # Store current state in state tracker array.
            states[j][0] = r
            states[j][1] = phi
            states[j][2] = vr
            states[j][3] = vphi
            states[j][4] = t

            E1 = e_func(phi, r, k_e)
            vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
            E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
            vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
            E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
            vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
            E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
            vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

            phif = phi + t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
            k_i, k_f = inKicker_func(phif % (2 * np.pi))
            if kick_counter < pseudo_kick_max and k_i != 0:
                r, phi, vr, vphi, t, lost_in_kicker, j = kicker_s(states, j, 
                    r_max, r_min, b_nom, b_k, t_list, k_i, k_f, k_e, e_func, t_step, t_f, exit_checker, inj_region)
                if lost_in_kicker:
                    lost = True
                    break
                elif t > t_f:
                    break
                kick_counter += 1
                continue

            r = r + t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
            phi = phif
            vr = vr + t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
            vphi = vphi + t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
            t = t + t_step
            j += 1
        return states[:j], lost

    @staticmethod
    @nb.njit(fastmath = True, parallel = True)
    def _jump_rk4_m(state, r_max, r_min, b_nom, b_k, t_list, pseudo_kick_max, inKicker_func, k_e, e_func,
        t_f, t_step, inj_region, exit_checker):

        lost = np.full(shape = len(state), fill_value = False)

        for i in nb.prange(len(state)):
            # Extract the initial state.
            r = state[i][0]
            phi = state[i][1]
            vr = state[i][2]
            vphi = state[i][3]
            t = state[i][4]
            
            kick_counter = 0

            while t < t_f:
                if phi > inj_region and not exit_checker(r, phi, t_step, r_max, r_min):
                    lost[i] = True
                    break

                E1 = e_func(phi, r, k_e)
                vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
                E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, k_e)
                vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
                E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, k_e)
                vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
                E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, k_e)
                vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

                phif = phi + t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
                k_i, k_f = inKicker_func(phif % (2 * np.pi))
                if kick_counter < pseudo_kick_max and k_i != 0:
                    r, phi, vr, vphi, t, lost_in_kicker = kicker_m(r, phi, vr, vphi, t, 
                        r_max, r_min, b_nom, b_k, t_list, k_i, k_f, k_e, e_func, t_step, t_f, exit_checker, inj_region)
                    if lost_in_kicker:
                        lost[i] = True
                        break
                    elif t > t_f:
                        break
                    kick_counter += 1
                    continue

                r = r + t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
                phi = phif
                vr = vr + t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
                vphi = vphi + t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
                t = t + t_step
            
            # Update the states array with the final state of the muon.
            state[i][0] = r 
            state[i][1] = phi 
            state[i][2] = vr 
            state[i][3] = vphi 
            state[i][4] = t
        return state, lost

    @staticmethod
    @nb.njit(fastmath = True, cache = True)
    def _jump_optical(state, r_max, r_min, b_nom, b_k, t_list, pseudo_kick_max, inKicker_func, k_e, e_func,
        t_f, t_step, inj_region, exit_checker):
        pass
        # split = 100
        # dtheta = 2 * np.pi / split
        # delta_p = 0.001
        # r_magic = 7.012

        # constant_matrix = np.array([
        #     aux.r_magic * delta_p * (1 - np.cos(dtheta)), 
        #     aux.r_magic * delta_p * np.sin(dtheta), 
        #     ((1 + delta_p) * dtheta - delta_p * np.sin(dtheta)) / aux.omega_magic
        # ])

        # transfer_matrix = np.array([
        #     [np.cos(dtheta), np.sin(dtheta), 0, 0],
        #     [-np.sin(dtheta), np.cos(dtheta), 0],
        #     [np.sin(dtheta) / v0, (1 - np.cos(dtheta)) / v0, 1]
        # ])