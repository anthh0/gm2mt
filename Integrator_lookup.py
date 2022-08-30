import numpy as np
import numba as nb
import math
import gm2mt.auxiliary as aux

m = aux.m # muon mass in kg
q = aux.q # muon charge in C
c = aux.c # speed of light in m/s

#region B AND E FIELD FUNCTIONS:
@nb.njit(nb.boolean(nb.float64), fastmath = True, cache = True)
def inQuad(angle):                                      # ring angle (NOT Cartesian angle)
    angle = angle % (2 * np.pi)
    if (angle > aux.Q1S_i and angle < aux.Q1L_f) or (angle > aux.Q2S_i and angle < aux.Q2L_f) or (angle > aux.Q3S_i and angle < aux.Q3L_f) or (angle > aux.Q4S_i and angle < aux.Q4L_f):
        return True                                     # returns boolean True when muon is in quad region
    else:
        return False
    
@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_1(angle):                   # returns the angular bounds of the occupied (single) kicker
    angle = angle % (2 * np.pi)
    if angle > aux.k1_i and angle < aux.k3_f_no_gaps:
        return aux.k1_i, aux.k3_f_no_gaps
    else:
        return 0.0, 0.0

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_3(angle):                   # returns the angular bounds of the occupied (three part) kicker
    angle = angle % (2 * np.pi)
    if angle > aux.k1_i and angle < aux.k1_f:
        return aux.k1_i, aux.k1_f
    elif angle > aux.k2_i and angle < aux.k2_f:
        return aux.k2_i, aux.k2_f
    elif angle > aux.k3_i and angle < aux.k3_f:
        return aux.k3_i, aux.k3_f
    else:
        return 0.0, 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64[:,:]), fastmath = True, cache = True)
def e_eval_1(angle, r, e_lookup):                                     # returns E_r linear approx for continuous ESQ plates
    if r > aux.r_magic - 0.05 and r < aux.r_magic + 0.05:
        return np.interp(r, e_lookup[0], e_lookup[1])
    else:
        e_str = 0.0
        return e_str

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64[:,:]), fastmath = True, cache = True)
def e_eval_4(angle, r, e_lookup):                                    # returns E_r linear approx for discrete ESQ plates
    if inQuad(angle) and r > aux.r_magic - 0.05 and r < aux.r_magic + 0.05:
        return np.interp(r, e_lookup[0], e_lookup[1])
    else:
        e_str = 0.0
        return e_str

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def cyl_deriv(r, vr, vphi, B, E):       # returns the r, phi components of the acceleration vector
    m = aux.m                                                                   # muon mass in kg
    q = aux.q                                                                   # muon charge in C
    c = aux.c                                                                   # speed of light in m/s
    gamma = 1 / np.sqrt(1 - (vr**2 + r**2*vphi**2)/c**2)
    ar = r*vphi**2 + q/gamma/m*(E - B*r*vphi - vr**2*E/c**2)                    # note that B = -B_z
    aphi = (q/gamma/m*(B*vr - E*r*vr*vphi/c**2) - 2*vr*vphi) / r
    return ar, aphi
#endregion

#region EXIT CHECKER FUNCTIONS:
@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_cont(r, phi, t_step, r_max, r_min):                              # continuous collimator exit checker
    if (r > r_max or r < r_min):
        return False                                                            # returns False if particle has exited the ring aperature
    else:
        return True                                                             # returns True if r is inside r_min and r_max

@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_disc(r, phi, t_step, r_max, r_min):                              # discrete collimator exit checker
    phi = phi % (2*np.pi)
    width = 2*np.pi * t_step/1.491E-7                       # Angular displacement in t_step amount of time -- approx vphi*t_step
    C1 = aux.C1                                             # Collimator angular locations
    C2 = aux.C2
    C3 = aux.C3
    C4 = aux.C4
    C5 = aux.C5
    if (abs(phi - C1) < width or abs(phi - C2) < width or abs(phi - C3) < width or abs(phi - C4) < width or abs(phi - C5) < width) and (r > r_max or r < r_min):
        return False                    # returns False if the particle is outside the ring aperture and within one t_step of a collimator
    else:                               # i don't think these should be abs-vals, a particle that's already past the collimator doesn't need to be discarded?
        return True                     # returns True if the particle is still in the ring and not near a collimator


@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_none(r, phi, t_step, r_max, r_min):
    return True                         # for no collimation; particle is never lost --> returns True
#endregion

#region RK4 KICKER FUNCTIONS:           --------------Here be an RK4 implementation---------------
@nb.njit(fastmath = True, cache = True)
def tstep_init(r, phi, vr, vphi, b_nom, target, e_lookup, e_func, original_t_step):      # doesn't take a t-input, so idk what to pass to updated_state_rk4
    jump_width = original_t_step / 10                                               # same goes for b_func
    t_step = jump_width
    while t_step <= original_t_step:

        E1 = e_func(phi, r, e_lookup)
        vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
        E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
        vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
        E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
        vphi3, aphi3 = vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)[1]

        vphi4 = vphi + t_step * aphi3                   # only concerned about phi here

        final_angle = phi + t_step*(vphi1 + 2*vphi2 + 2*vphi3 + vphi4)/6
        if final_angle > target:                        # once the final phi is past the target stop incrementing t_step
            break
        else:
            t_step += jump_width
    return t_step                                       # return the t_step that got closest to the target

@nb.njit(fastmath = True, cache = True)
def kicker_s(states, j, r_max, r_min, b_nom, b_k, t_list, k_i, k_f, e_lookup, e_func, t_step, t_f, exit_checker, inj_region):
    # EDGE MATCHING, first iteration - match the timestep to the leading kicker edge
    r = states[j][0]                                    # get the ICs from the jth state (j is temporal index i think)
    phi = states[j][1]
    vr = states[j][2]
    vphi = states[j][3]
    t = states[j][4]
    
    original_t_step = t_step                            # use tstep_init to find largest t_step not yet in the kicker region
    t_step = tstep_init(r, phi % (2 * np.pi), vr, vphi, b_nom, k_i, e_lookup, e_func, original_t_step)
    
    E1 = e_func(phi, r, e_lookup)                            # RK4 method in action, B = b_nom when not in the kicker regions
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step                                         # use RK4 method to calculate next intermediary state

    # EDGE MATCHING, second iteration - fill out the rest of the timestep
    t_step = original_t_step - t_step                   # remainder of the time step (in the kicker region)
                                                        # calculate the kicker B fields through interp against the exp data (if t_list, b_k are exp data that is)
    B1, B2, B3, B4 = b_nom - np.interp(t, t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + t_step, t_list, b_k)

    E1 = e_func(phi, r, e_lookup)                            # RK4 method
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step                                         # RK4 method to calculate next intermediary state
    j += 1

    states[j][0] = r                                    # update state tracker
    states[j][1] = phi
    states[j][2] = vr
    states[j][3] = vphi
    states[j][4] = t

    # KICKER PATCHING: finely propagate through the kicker region, with "split" defining the fineness;
    #                  loop ends when it reaches the sim lifetime end or exits the kicker
    split = 5
    t_list_fine = np.arange(t, t + 2E-8, original_t_step / (2 * split) ) # where does 2E-8 come from?
    b_k_fine = np.interp(t_list_fine, t_list, b_k)      # kicker B field contribution, note evaluated at twice the number of points (half the spacing) to allow indexing at t_step/2 times
    t_step = original_t_step / split                    # t_step is now smol t_step
    kicker_idx = 0
    lost_in_kicker = False

    while t < t_f and (phi % (2 * np.pi)) < k_f:        # in kicker and prior to sim end
        if phi > inj_region and not exit_checker(r, phi, original_t_step, r_max, r_min):
            lost_in_kicker = True                       # nooooo poor muons they're sad
            break

        states[j][0] = r                                # load initial (updated) state
        states[j][1] = phi
        states[j][2] = vr
        states[j][3] = vphi
        states[j][4] = t

        for i in nb.prange(split):                      # prange() doesn't execute strictly in order -- is that important?
            B1 = b_nom - b_k_fine[2 * kicker_idx]       # RK4 again bb
            E1 = e_func(phi, r, e_lookup)
            vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)

            B2 = b_nom - b_k_fine[2 * kicker_idx + 1]   # note indexing b_k_fine at a +t_step/2 time
            E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
            vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)

            B3 = b_nom - b_k_fine[2 * kicker_idx + 1]
            E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
            vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)

            B4 = b_nom - b_k_fine[2 * kicker_idx + 2]   # note indexing b_k_fine at a +t_step time
            E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
            vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

            r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
            phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
            vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
            vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
            t += t_step
            kicker_idx += 1                             # update intermediary state and b_k_fine index

            if (phi % (2 * np.pi)) > k_f:               # if muon leaves the kicker region (while loop will terminate on next iteration)
                t_step = (split - i) * t_step           # t_step is now medium-sized t_step

                E1 = e_func(phi, r, e_lookup)                # oh yeah RK4 time (could prolly make this its own function lol)
                vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)

                E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
                vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)

                E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
                vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)

                E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
                vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

                r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
                phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
                vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
                vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
                t += t_step                             # remaining state update

                break                                   # idk if this is needed (it may be detrimental, see below)
        
        j += 1                                          # would j update if muon exits the kicker region?
    
    return r, phi, vr, vphi, t, lost_in_kicker, j       # return ultimate state at sim end or upon leaving kicker

@nb.njit(fastmath = True, cache = True)                 # 18 inputs :() -- Note state tracker and state index j are not included
def kicker_m(r, phi, vr, vphi, t, r_max, r_min, b_nom, b_k, t_list, k_i, k_f, e_lookup, e_func, t_step, t_f, exit_checker, inj_region):
    # EDGE MATCHING, first iteration - match the timestep to the leading kicker edge
    
    original_t_step = t_step                            # returns the t_step size to get closest to the kicker edge
    t_step = tstep_init(r, phi % (2 * np.pi), vr, vphi, b_nom, k_i, e_lookup, e_func, original_t_step)
    E1 = e_func(phi, r, e_lookup)                            # RK4 implementation -- outside the kicker region B = b_nom
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6   #update intermediary state with calculated values
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step

    # EDGE MATCHING, second iteration - fill out the rest of the timestep

    t_step = original_t_step - t_step                   # remainder of the first t_step -- inside the kicker region
    
    B1, B2, B3, B4 = b_nom - np.interp(t, t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + (t_step / 2), t_list, b_k), b_nom - np.interp(t + t_step, t_list, b_k)

    E1 = e_func(phi, r, e_lookup)                            # RK4 it up -- using interped kicker B field
    vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)
    E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
    vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)
    E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
    vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)
    E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
    vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

    r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6   # update the intermediary state again
    phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
    vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
    vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
    t += t_step                                         # no j index to update

    # KICKER PATCHING: finely propagate through the kicker region, with "split" defining the fineness;
    #                  loop ends when it reaches the sim lifetime end or exits the kicker
    
    split = 5
    t_list_fine = np.arange(t, t + 2E-8, original_t_step / (2 * split) )        # still don't know where 2E-8 comes from???
    b_k_fine = np.interp(t_list_fine, t_list, b_k)      # kicker B field at half smol t_steps
    t_step = original_t_step / split                    # smol t_steps
    kicker_idx = 0          
    lost_in_kicker = False

    while t < t_f and (phi % (2 * np.pi)) < k_f:        # while prior to sim end and still in kicker region
        if phi > inj_region and not exit_checker(r, phi, original_t_step, r_max, r_min):
            lost_in_kicker = True                       # nooooo sad muon
            break

        B1 = b_nom - b_k_fine[2 * kicker_idx]           # note init step index
        E1 = e_func(phi, r, e_lookup)
        vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, B1, E1)

        B2 = b_nom - b_k_fine[2 * kicker_idx + 1]       # note half step index
        E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
        vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, B2, E2)

        B3 = b_nom - b_k_fine[2 * kicker_idx + 1]
        E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
        vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, B3, E3)

        B4 = b_nom - b_k_fine[2 * kicker_idx + 2]       # note full step index
        E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
        vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, B4, E4)

        r += t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
        phi += t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
        vr += t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
        vphi += t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
        t += t_step
        kicker_idx += 1                                 # update the state and kicker index
    
    return r, phi, vr, vphi, t, lost_in_kicker          # this func (kicker_m) seems less error prone than kicker_s
#endregion

class Integrator:
    def __init__(self, integration_method):
        self.integration_method = integration_method
    
    def integrate(self, init_states, ring, dt, t_f):        # returns the intitialized integrator_func
        dt /= 10**9                                         # convert to nanoseconds
        t_f /= 10**9
        inj_region = aux.C1                                 # location of C1 (rad)
        # kicker_turns = ring.b_k.kick_max
        # if ring.b_k.kicker_num == 3:
        #     inj_region = (2 * np.pi * (kicker_turns-1)) + aux.k3_f
        # if ring.b_k.kicker_num == 1:
        #     inj_region = (2 * np.pi * (kicker_turns-1)) + aux.k3_f_no_gaps

        if ring.quad_num == 1:                              # choose the quad model
            e_func = e_eval_1
        else:
            e_func = e_eval_4
        
        # if ring.quad_model == "linear":
        #     e_lookup = ring.e_linear_lookup                 # choose the E field lookup table
        # elif ring.quad_model == "full":
        #     e_lookup = ring.e_full_lookup

        if ring.collimators == "continuous":                # choose the collimator model
            exit_checker = exit_check_cont
        elif ring.collimators == "discrete":
            exit_checker = exit_check_disc
        else:
            exit_checker = exit_check_none
        
        if ring.b_k.kicker_num == 1:                        # choose the kicker model
            inKicker_func = inKicker_1
        elif ring.b_k.kicker_num == 3:
            inKicker_func = inKicker_3
        else:
            raise ValueError("kickernum unrecognized")

        init_states_c = init_states.copy() # original init_state should NEVER be touched; do any operations on this copied version
        # print('Shape of init_state_c:', init_states_c.shape)
        if len(init_states_c) == 1:
            init_states_c = init_states_c[0]
            times = np.arange(init_states_c[4], t_f, dt)
            b_k = np.interp(times, (ring.b_k).t_list, (ring.b_k).b_k)
            if self.integration_method == "rk4":            # choose the integrator model
                integrator_func = Integrator._jump_rk4_s
            elif self.integration_method == "rkn":
                print("Working on it")
            elif self.integration_method == "optical":
                pass
            # RETURNS: states array, lost
        else:
            times = ring.b_k.t_list
            b_k = ring.b_k.b_k
            if self.integration_method == "rk4":
                integrator_func = Integrator._jump_rk4_m
            elif self.integration_method == "rkn":
                print("Working on it \o_o/")
            elif self.integration_method == "optical":
                pass
            # RETURNS: final_states, lost array
        
        return integrator_func(                             # initialize the integrator_func and return it
            state = init_states_c, # a NumPy array for the "in"-state
            r_max = ring.r_max,
            r_min = ring.r_min,
            b_nom = ring.b_nom,
            b_k = b_k,
            t_list = times,
            pseudo_kick_max = ring.b_k.kick_max * ring.b_k.kicker_num, # a pass through the three kickers is one true kick but three pseudo kicks
            inKicker_func = inKicker_func,
            e_lookup = ring.e_lookup,
            e_func = e_func,
            t_f = t_f,
            t_step = dt,
            inj_region = inj_region,
            exit_checker = exit_checker)

    # Here are the true 'integration' functions
    @staticmethod
    @nb.njit(fastmath = True, cache = True)                 # for single muons
    def _jump_rk4_s(state, r_max, r_min, b_nom, b_k, t_list, pseudo_kick_max, inKicker_func, e_lookup, e_func,
        t_f, t_step, inj_region, exit_checker):
        r = state[0]
        phi = state[1]
        vr = state[2]
        vphi = state[3]
        t = state[4]

        max_jumps = int( (t_f - t) // t_step) + 1
        states = np.empty(shape = (max_jumps, 5))           # initialize an empty state tracker array
        kick_counter, j = 0, 0
        lost = False
        while j < max_jumps:
            if phi > inj_region and not exit_checker(r, phi, t_step, r_max, r_min):
                lost = True                                 # sad muon
                break
            
            # Store current state in state tracker array.
            states[j][0] = r
            states[j][1] = phi
            states[j][2] = vr
            states[j][3] = vphi
            states[j][4] = t

            E1 = e_func(phi, r, e_lookup)                        # RK4 -- assumed not in kicker region
            vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
            E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
            vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
            E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
            vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
            E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
            vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

            phif = phi + t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
            k_i, k_f = inKicker_func(phif % (2 * np.pi))    # if muon would land in a kicker return the angular start, finish
            if kick_counter < pseudo_kick_max and k_i != 0: # in that case use kicker_s to integrate over the boundary
                r, phi, vr, vphi, t, lost_in_kicker, j = kicker_s(states, j, 
                    r_max, r_min, b_nom, b_k, t_list, k_i, k_f, e_lookup, e_func, t_step, t_f, exit_checker, inj_region)
                if lost_in_kicker:
                    lost = True                             # sad muon
                    break
                elif t > t_f:
                    break
                kick_counter += 1                           # increase kick count
                continue

            r = r + t_step * (vr1 + 2 * vr2 + 2 * vr3 + vr4) / 6
            phi = phif
            vr = vr + t_step * (ar1 + 2 * ar2 + 2 * ar3 + ar4) / 6
            vphi = vphi + t_step * (aphi1 + 2 * aphi2 + 2 * aphi3 + aphi4) / 6
            t = t + t_step                                  # isn't this already completed in kicker_s?
            j += 1
        return states[:j], lost

    @staticmethod
    @nb.njit(fastmath = True, parallel = True)              # for multiple muons
    def _jump_rk4_m(state, r_max, r_min, b_nom, b_k, t_list, pseudo_kick_max, inKicker_func, e_lookup, e_func,
        t_f, t_step, inj_region, exit_checker):

        lost = np.full(shape = len(state), fill_value = False)  # big array of False

        for i in nb.prange(len(state)):                     # i don't think nb.prange() generally executes in order
            # Extract the initial state.                    # i is the muon index, so execution order isn't important
            r = state[i][0]
            phi = state[i][1]
            vr = state[i][2]
            vphi = state[i][3]
            t = state[i][4]
            
            kick_counter = 0

            while t < t_f:
                if phi > inj_region and not exit_checker(r, phi, t_step, r_max, r_min):
                    lost[i] = True                          # sad muon
                    break

                E1 = e_func(phi, r, e_lookup)                    # RK4 to check if muon would land in kicker region
                vr1, vphi1, (ar1, aphi1) = vr, vphi, cyl_deriv(r, vr, vphi, b_nom, E1)
                E2 = e_func(phi + t_step / 2 * vphi1, r + t_step / 2 * vr1, e_lookup)
                vr2, vphi2, (ar2, aphi2) = vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, cyl_deriv(r + t_step / 2 * vr1, vr + t_step / 2 * ar1, vphi + t_step / 2 * aphi1, b_nom, E2)
                E3 = e_func(phi + t_step / 2 * vphi2, r + t_step / 2 * vr2, e_lookup)
                vr3, vphi3, (ar3, aphi3) = vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, cyl_deriv(r + t_step / 2 * vr2, vr + t_step / 2 * ar2, vphi + t_step / 2 * aphi2, b_nom, E3)
                E4 = e_func(phi + t_step * vphi3, r + t_step * vr3, e_lookup)
                vr4, vphi4, (ar4, aphi4) = vr + t_step * ar3, vphi + t_step * aphi3, cyl_deriv(r + t_step * vr3, vr + t_step * ar3, vphi + t_step * aphi3, b_nom, E4)

                phif = phi + t_step * (vphi1 + 2 * vphi2 + 2 * vphi3 + vphi4) / 6
                k_i, k_f = inKicker_func(phif % (2 * np.pi))
                if kick_counter < pseudo_kick_max and k_i != 0:     # propagate through kicker region with kicker_m
                    r, phi, vr, vphi, t, lost_in_kicker = kicker_m(r, phi, vr, vphi, t, 
                        r_max, r_min, b_nom, b_k, t_list, k_i, k_f, e_lookup, e_func, t_step, t_f, exit_checker, inj_region)
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
                t = t + t_step                                      # check to see if this occurs in kicker_m?
            
            # Update the states array with the final state of the muon.
            state[i][0] = r 
            state[i][1] = phi 
            state[i][2] = vr 
            state[i][3] = vphi 
            state[i][4] = t
        return state, lost


