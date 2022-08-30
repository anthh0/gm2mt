import numpy as np
import numba as nb
import math
import gm2mt.auxiliary as aux

m = aux.m                                                                           # muon mass in kg
q = aux.q                                                                           # muon charge in C
c = aux.c                                                                           # speed of light in m/s

#region B AND E FIELD FUNCTIONS:
@nb.njit(nb.boolean(nb.float64), fastmath = True, cache = True)
def inQuad_1(angle):                                                                 # muon is always in continous ESQ plates
    return True

@nb.njit(nb.boolean(nb.float64), fastmath = True, cache = True)
def inQuad_4(angle):                                                                 # returns boolean True if muon is in quad region #                
    angle = angle % (2 * np.pi)                                                      # ring angle (NOT Cartesian angle)
    if (angle > aux.Q1S_i and angle < aux.Q1L_f) or (angle > aux.Q2S_i and angle < aux.Q2L_f) or (angle > aux.Q3S_i and angle < aux.Q3L_f) or (angle > aux.Q4S_i and angle < aux.Q4L_f):
        return True                                    
    else:
        return False
    
@nb.njit(nb.types.Tuple((nb.boolean, nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_1(angle):                                                              # returns boolean and the angular bounds of the occupied (combined) kicker
    angle = angle % (2 * np.pi)
    if angle > aux.k1_i and angle < aux.k3_f_no_gaps:
        return True, aux.k1_i, aux.k3_f_no_gaps
    else:
        return False, 0.0, 0.0

@nb.njit(nb.types.Tuple((nb.boolean, nb.float64, nb.float64))(nb.float64), fastmath = True, cache = True)
def inKicker_3(angle):                                                              # returns boolean and the angular bounds of the occupied (three part) kicker
    angle = angle % (2 * np.pi)
    if angle > aux.k1_i and angle < aux.k1_f:
        return True, aux.k1_i, aux.k1_f
    elif angle > aux.k2_i and angle < aux.k2_f:
        return True, aux.k2_i, aux.k2_f
    elif angle > aux.k3_i and angle < aux.k3_f:
        return True, aux.k3_i, aux.k3_f
    else:
        return False, 0.0, 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_linear_1(angle, r, k_e):
    if r > 7.062 and r < 7.162:
        return (k_e * (r - 7.112))
    else:
        e_str = 0.0
        return e_str

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_linear_4(angle, r, k_e):
    if inQuad_4(angle) and r > 7.062 and r < 7.162:
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
    if inQuad_4(angle) and r > 7.062 and r < 7.162:
        r_o = r - 7.112                     # r offset from r_magic [m]
        return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
    else:
        return 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def e_eval_full_exterior_4(angle, r, k_e):
    s = 0.1
    w = 0.047
    sig = k_e*(s**2 + w**2)/32/w
    if inQuad_4(angle):
        r_o = r - 7.112                     # r offset from r_magic [m]
        return 2*sig*(np.log((r_o + w/2)**2 + (s/2)**2) - np.log((r_o - w/2)**2 + (s/2)**2)) - 4*sig*(np.arctan(w/2/(r_o - s/2)) + np.arctan(w/2/(r_o + s/2)))
    else:
        return 0.0

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64[:]), fastmath = True, cache = True)
def b_eval_1(phi, t, b_t0, b_k):                                                    # returns interpolated approx of B for single kicker
    if inKicker_1(phi)[0]:
        i = int((t - b_t0)//1.25)                                                   # the entries in run2 kicker pulse are (around) 1.25ns apart
        bk = b_k[i] + ((t-b_t0) % 1.25)/1.25 * (b_k[i+1] - b_k[i])
        return aux.B_nom - bk
    else:
        return aux.B_nom

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64[:]), fastmath = True, cache = True)
def b_eval_3(phi, t, b_t0, b_k):                                                    # returns interpolated approx of B for (realistic) triple kicker
    if inKicker_3(phi)[0]:
        i = int((t - b_t0)//1.25)                                                   # the entries in run2 kicker pulse are (around) 1.25ns apart
        bk = b_k[i] + ((t-b_t0) % 1.25)/1.25 * (b_k[i+1] - b_k[i])
        return aux.B_nom - bk
    else:
        return aux.B_nom

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def cyl_acc(r, vr, vphi, B, E):                                                     # returns the acceleration in cyl coords given pos, vels and fields
    m = 1.883531627E-28                                                             # muon mass in kg
    q = 1.602176634E-19                                                             # muon charge in C
    c = 299792458                                                                   # speed of light in m/s
    gamma = 1 / np.sqrt(1 - (vr**2 + r**2*vphi**2)/c**2)
    ar = r*vphi**2 + q/gamma/m*(E - B*r*vphi - vr**2*E/c**2)                        # note that B = -B_z
    aphi = (q/gamma/m*(B*vr - E*r*vr*vphi/c**2) - 2*vr*vphi) / r
    return ar, aphi
#endregion

#region STATE UPDATING FUNCTIONS:
@nb.njit(fastmath = True, cache = True)
def next_state_rk4(init_state, dt, e_func, k_e, b_func, b_t0, b_k):
    
    r0, phi0, vr0, vphi0, t0 = init_state
    ar0, aphi0 = cyl_acc(r0, vr0, vphi0, b_func(phi0, t0, b_t0, b_k), e_func(r0, phi0, k_e))

    r1, phi1 = r0 + dt/2*vr0, phi0 + dt/2*vphi0
    vr1, vphi1 = vr0 + dt/2*ar0, vphi0 + dt/2*aphi0
    ar1, aphi1 = cyl_acc(r1, vr1, vphi1, b_func(phi1, t0 + dt/2, b_t0, b_k), e_func(r1, phi1, k_e))

    r2, phi2 = r0 + dt/2*vr1, phi0 + dt/2*vphi1
    vr2, vphi2 = vr0 + dt/2*ar1, vphi0 + dt/2*aphi1
    ar2, aphi2 = cyl_acc(r2, vr2, vphi2, b_func(phi2, t0 + dt/2, b_t0, b_k), e_func(r2, phi2, k_e))

    r3, phi3 = r0 + dt*vr2, phi0 + dt*vphi2
    vr3, vphi3 = vr0 + dt*ar2, vphi0 + dt*aphi2
    ar3, aphi3 = cyl_acc(r3, vr3, vphi3, b_func(phi3, t0 + dt, b_t0, b_k), e_func(r3, phi3, k_e))

    r_new, phi_new = r0 + dt*(vr0 + 2*vr1 + 2*vr2 + vr3)/6, phi0 + dt*(vphi0 + 2*vphi1 + 2*vphi2 + vphi3)/6
    vr_new, vphi_new = vr0 + dt*(ar0 + 2*ar1 + 2*ar2 + ar3)/6, vphi0 + dt*(aphi0 + 2*aphi1 + 2*aphi2 + aphi3)/6

    return np.array([r_new, phi_new, vr_new, vphi_new, t0 + dt])

@nb.njit(fastmath = True, cache = True)
def next_state_rkn(init_state, dt, e_func, k_e, b_func, b_t0, b_k):                              # PROTOTYPE #
    
    r0, phi0, vr0, vphi0, t0 = init_state
    ar0, aphi0 = cyl_acc(r0, vr0, vphi0, b_func(phi0, t0, b_t0, b_k), e_func(r0, phi0, k_e))

    vr1, vphi1 = vr0 + dt/2*ar0, vphi0 + dt/2*aphi0
    r1, phi1 = r0 + dt/2*(4*vr0 + 2*vr1 + dt/2*ar0)/6, phi0 + dt/2*(4*vphi0 + 2*vphi1 + dt/2*aphi0)/6
    ar1, aphi1 = cyl_acc(r1, vr1, vphi1, b_func(phi1, t0 + dt/2, b_t0, b_k), e_func(r1, phi1, k_e))

    vr2, vphi2 = vr0 + dt/2*ar1, vphi0 + dt/2*aphi1
    r2, phi2 = r0 + dt/2*(4*vr0 + 2*vr2 + dt/2*ar0)/6, phi0 + dt/2*(4*vphi0 + 2*vphi2 + dt/2*aphi0)/6
    ar2, aphi2 = cyl_acc(r2, vr2, vphi2, b_func(phi2, t0 + dt/2, b_t0, b_k), e_func(r2, phi2, k_e))

    vr3, vphi3 = vr0 + dt*ar2, vphi0 + dt*aphi2
    r3, phi3 = r0 + dt*(4*vr0 + 2*vr3 + dt*ar0)/6, phi0 + dt*(4*vphi0 + 2*vphi3 + dt*aphi0)/6
    ar3, aphi3 = cyl_acc(r3, vr3, vphi3, b_func(phi3, t0 + dt, b_t0, b_k), e_func(r3, phi3, k_e))

    r_new, phi_new = r0 + dt*(vr0 + 2*vr1 + 2*vr2 + vr3)/6, phi0 + dt*(vphi0 + 2*vphi1 + 2*vphi2 + vphi3)/6
    vr_new, vphi_new = vr0 + dt*(ar0 + 2*ar1 + 2*ar2 + ar3)/6, vphi0 + dt*(aphi0 + 2*aphi1 + 2*aphi2 + aphi3)/6

    return np.array([r_new, phi_new, vr_new, vphi_new, t0 + dt])
#endregion

#region EXIT CHECKER FUNCTIONS:
@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_cont(r, phi, t_step):                                                # continuous collimator exit checker
    if (r > 7.157 or r < 7.067):
        return False                                                                # returns False if particle has exited the ring aperature
    else:
        return True                                                                 # returns True if r is inside r_min and r_max

@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_disc(r, phi, t_step):                                                # discrete collimator exit checker
    phi = phi % (2*np.pi)
    width = 2*np.pi * t_step/1.491E-7                                               # Angular displacement in t_step amount of time -- approx vphi*t_step
    C1 = 2.217941507311586                                                          # collimator locations (rad)
    C2 = 2.7065523059617544
    C3 = 4.277418447694038
    C4 = 5.359533183352081
    C5 = 5.84814398200225
    phi1, phi2, phi3, phi4, phi5 = C1 - phi, C2 - phi, C3 - phi, C4 - phi, C5 - phi # Angular distance to each collimator
    if ((phi1 > 0 and phi1 < width) or (phi2 > 0 and phi2 < width) or  (phi3 > 0 and phi3 < width) or  (phi4 > 0 and phi4 < width) or  (phi5 > 0 and phi5 < width)) and (r < 7.067 or r > 7.157):
        return False                                                                # Returns False if muon is passing a collimator and outside the ring
    else:
        return True

@nb.njit(nb.boolean(nb.float64, nb.float64, nb.float64), fastmath = True, cache = True)
def exit_check_none(r, phi, t_step):
    return True                                                                     # for no collimation; particle is never lost --> returns True
#endregion

#region KICKER FUNCTIONS:
@nb.jit(fastmath = True, cache = True)
def tstep_init(init_state, target, original_t_step, update_func, e_func, k_e, b_func, b_t0, b_k):
    coarse_jump_width = original_t_step / 10
    fine_jump_width = original_t_step / 100
    t_step = 0                                                                      # start with smol (zero) t_step
    
    while t_step < original_t_step:

        final_angle = update_func(init_state, t_step, e_func, k_e, b_func, b_t0, b_k)[1] # where the muon would go in smol t_step

        if final_angle > target:
            break
        else:
            t_step += coarse_jump_width
    
    # while t_step < original_t_step:                                                 # secondary iteration to get more precise t_step
        
    #     final_angle = update_func(init_state, t_step, e_func, k_e, b_func, b_t0, b_k)[1]
        
    #     if final_angle > target:
    #         break
    #     else:
    #         t_step += fine_jump_width  
    
    return t_step                                                                   # return the t_step that got closest to the target without passing it


class Integrator:
    def __init__(self, integration_method):
        if integration_method in ['rk4', 'rkn']:
            self.integration_method = integration_method
        else:
            raise ValueError(f"Your integration method: {integration_method} is not a supported option ¯\_(ツ)_/¯")

    def integrate(self, init_states, ring, dt, t_f):
        self.dt = dt / 1E9                                                          # convert to nanoseconds
        self.t_f =  t_f / 1E9
        self.inj_region = aux.C1

        if ring.collimators == "continuous":                                        # select the appropriate collimator model
            exit_checker = exit_check_cont
        elif ring.collimators == "discrete":
            exit_checker = exit_check_disc
        else:
            exit_checker = exit_check_none

        if ring.quad_num == 1:                                                      # select the appropriate ESQ field evaluator
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

        if ring.b_k.kicker_num == 1:                                                # select the appropriate kicker checker and b_eval model
            b_func = b_eval_1
            kicker_checker = inKicker_1
        else:
            b_func = b_eval_3
            kicker_checker = inKicker_3

        if self.integration_method == 'rk4':                                        # select the appropriate state updater func
            update_func = next_state_rk4
        elif self.integration_method == 'rkn':
            update_func = next_state_rkn

        init_states_c = init_states.copy()                                          # original initial states should NEVER be touched, shape is [muon num] x [r, phi, vr, vphi, t]

        return Integrator._compute_final_states(
            t_step = self.dt,
            t_f = self.t_f,
            init_states = init_states_c,
            exit_checker = exit_checker,
            kicker_checker = kicker_checker,
            update_func = update_func,
            e_func = e_func,
            k_e = ring.k_e,
            b_func = b_func,
            b_t0 = ring.b_k.t_list[0],
            b_k = ring.b_k.b_k)

    @staticmethod
    def _compute_final_states(t_step, t_f, init_states, exit_checker, kicker_checker, update_func, e_func, k_e, b_func, b_t0, b_k):
        if len(init_states) == 1:                                                   # single muon sims
            return Integrator._compute_final_states_s(t_step, t_f, init_states, exit_checker, kicker_checker, update_func, e_func, k_e, b_func, b_t0, b_k)
        else:                                                                       # multiple muon sims
            return Integrator._compute_final_states_m(t_step, t_f, init_states, exit_checker, kicker_checker, update_func, e_func, k_e, b_func, b_t0, b_k)
    
    @staticmethod
    # @nb.njit(fastmath = True)                                                       # single muon sims track the state of the particle at every step of the sim
    def _compute_final_states_s(t_step, t_f, init_states, exit_checker, kicker_checker, update_func, e_func, k_e, b_func, b_t0, b_k):
        
        # still need to incorporate kick_counter and pseudo_kick_max (support for those options should be dropped imo but still)
        # state = list(init_states[0].copy())
        state = init_states[0]
        state_tracker = np.array([state])                                           # don't know how big this will get :/

        r, phi, vr, vphi, t = state                                                 # load the inital state
        lost = False
        in_kicker = False
        kick_counter = 0

        while t < t_f:
            t_rem = t_f - t
            t_step_min = min(t_step, t_rem)

            if phi > aux.C1 and not exit_checker(r, phi, t_step_min):
                lost = True                                                         # salute our fallen warrior o7
                break

            if not in_kicker:
                next_state = update_func(state, t_step_min, e_func, k_e, b_func, b_t0, b_k) # where the muon would land in a normal t_step
                kicker_check = kicker_checker(next_state[1])

                if kicker_check[0]:                                                 # if it would land in a kicker repeat with smol_t_step
                    smol_t_step = tstep_init(state, kicker_check[1], t_step_min, update_func, e_func, k_e, b_func, b_t0, b_k)
                    next_state = update_func(state, smol_t_step, e_func, k_e, b_func, b_t0, b_k)
                    in_kicker = True                                                # next update will be in a kicker region

                # state_tracker = np.append(state_tracker, np.array([next_state]), axis = 0) # save the next state to the state tracker
                state_tracker = np.append(state_tracker, [next_state], axis = 0)
                state = next_state
            else:
                split = 10
                fine_t_step = t_step / split                                        # finely propagate through the kicker region
                fine_t_step_min = min(fine_t_step, t_rem)

                next_state = update_func(state, fine_t_step_min, e_func, k_e, b_func, b_t0, b_k)
                print(f"phi: {phi}, t: {t}, B: {aux.B_nom - b_eval_3(phi, t, b_t0, b_k)}")
                kicker_check = kicker_checker(next_state[1])

                if not kicker_check[0]:                                             # if the muon would exit the kicker the next update will be regular
                    in_kicker = False

                # state_tracker = np.append(state_tracker, np.array([next_state]), axis = 0) # either way save the next state to the state tracker
                state_tracker = np.append(state_tracker, [next_state], axis = 0)
                state = next_state
                
            r, phi, vr, vphi, t = state

        print("Single Muon Integration Complete")
        return state_tracker, lost                                                  # return the state tracker array and whether the muon was lost

    @staticmethod
    @nb.njit(fastmath = True, parallel = True)                                      # multi muon sims only track the final states of each muon
    def _compute_final_states_m(t_step, t_f, init_states, exit_checker, kicker_checker, update_func, e_func, k_e, b_func, b_t0, b_k):
        states = init_states.copy()
        final_states = np.zeros(states.shape)                                       # multi muon sims only track the final state of each muon 
        lost = np.full(shape = len(states), fill_value = False)  
        
        for i in nb.prange(len(states)):
            r, phi, vr, vphi, t = states[i]
            in_kicker = False
            kick_counter = 0

            while t < t_f:
                t_rem = t_f - t
                t_step_min = min(t_step, t_rem)                                     # set the time step to the smaller of the normal step and the remaining sim time
                if phi > aux.C1 and not exit_checker(r, phi, t_step_min):
                    lost[i] = True                                                  # salute our fallen soldier o7
                    break
                        
                if not in_kicker:
                    next_state = update_func(states[i], t_step_min, e_func, k_e, b_func, b_t0, b_k)    # next state after normal-step propagation
                    kicker_check = kicker_checker(next_state[1])
                    
                    if kicker_check[0]:                                             # if it would take the muon into a kicker region, repeat propagation with small-step
                        smol_t_step = tstep_init(states[i], kicker_check[1], t_step_min, update_func, e_func, k_e, b_func, b_t0, b_k)
                        next_state = update_func(states[i], smol_t_step, e_func, k_e, b_func, b_t0, b_k)
                        in_kicker = True  
                        
                    states[i] = next_state                                          # either way save the next state
                else:
                    split = 10                                                      # finely propagate through kicker region with split defining the fineness
                    fine_t_step = t_step / split
                    fine_t_step_min = min(fine_t_step, t_rem)
                        
                    next_state = update_func(states[i], fine_t_step_min, e_func, k_e, b_func, b_t0, b_k)
                    kicker_check = kicker_checker(next_state[1])
                        
                    if not kicker_check[0]:                                         # if muon is no longer in kicker region, next iteration will be regular
                        in_kicker = False
                    states[i] = next_state                                          # either way save fine-step next state
                    
                r, phi, vr, vphi, t = states[i]

            final_states[i] = states[i]
            
        print("Multiple Muon Integration Complete")
        return final_states, lost
