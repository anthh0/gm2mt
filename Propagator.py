import numpy as np
import ROOT
import root_numpy as rnp
import scipy.optimize as opt

import gm2mt.auxiliary as aux
from gm2mt.Plotter import Plotter, AlignPlotter
from gm2mt.Integrator_OG import Integrator             # currently using the OG file for Integrator #
from gm2mt.StateGenerator import StateGenerator
import gm2mt.Distributions as dist
from gm2mt.Ring import Ring, _MRing
import gm2fr.constants as const

import time
import pathlib
import warnings
import pandas as pd
import os
import sys
import shutil
import contextlib


def _suppress_print(func):
    """Decorator for the .propagate method to suppress terminal printing."""
    def deco(*args, **kwargs):
        if args[0].suppress_print:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                value = func(*args, **kwargs)
        else:
            value = func(*args, **kwargs)
        return value
    return deco

class Propagator:
    """Controls and coordinates muon propagation, from initial condition simulation to plotting.

    The Propagator is a upper-level object that acts as a controller object, guiding the simulation process.
    It begins with MC generation of muon initial conditions, muon propagation, results and parameter storage, and plotting.

    It also contains several helper methods, which are not for external use.  For details, see their docstrings.
    ...

    ATTRIBUTES -------
    output : str / None
        name of the output directory, which will be prepended by "s-" or "m-" depending on single or multiple muon propagation
    display_options_s : list / str
        list of the plots to be shown in an interactive window after propagation; "_s" denotes the options for single muons, "_m" for multiple
    display_options_m : list / str
        see display_options_s
    compare_dir : str
        directory name for results of a momentum acceptance model simulation, for comparison on MT plots
    plot_option : str
        graph option for plot and font sizes, for detail or presentation (q.v. docstring of 'setting' method of Plotter class)
    animate : bool
        determines generation of muon path animation (only applicable in single-muon simulations)
    store_root : bool
        determiness generation of ROOT histogram of MT distribution (only applicable in multi-muon simulations)
    suppress_print : bool
        determines suppression of print statements to terminal
    integration_method : str
        method of solving the Lorentz equations of motion
    sim_time : float
        real-time length in seconds for simulation (not including plotting/saving)
    lost_count : int
        number of lost muons

    PUBLIC METHODS -------
    propagate(state_generator, ring, integration_method = "rk4", dt = 0.1, t_f = 1000):
        General method for muon propagation, results saving, and plot generation.
    """

    def __init__(
        self,
        output = 'testing', # directory for the output plots and .npz files
        plot = True,
        display_options_s = [],
        display_options_m = [],
        p_acceptance_dir = None,
        plot_option = "presentation",
        animate = False,
        store_root = False,
        store_npz = True,
        suppress_print = False,
        store_f_dist = True,
        # optimize_t_offset = False,                  # ADDED #
        multiplex_plots = []
    ):
        if isinstance(output, str):
            self.output = output
            print(f"Working into directory '/{output}.")
        else:
            raise ValueError("Your output directory is not recognized >:(")
        
        if isinstance(plot, bool):
            self.plot = plot
        else:
            raise TypeError("Your plot parameter is not recognized >:(")

        if isinstance(display_options_s, list):
            self.display_options_s = display_options_s
        elif isinstance(display_options_s, str):
            self.display_options_s = [display_options_s]
        else:
            raise TypeError("Please put your single-muon display options into a list or string >:(")

        if isinstance(display_options_m, list):
            self.display_options_m = display_options_m
        elif isinstance(display_options_m, str):
            self.display_options_m = [display_options_m]
        else:
            raise TypeError("Please put your multi-muon display options into a list or string >:(")

        if isinstance(p_acceptance_dir, str) or p_acceptance_dir is None:
            self.p_acceptance_dir = p_acceptance_dir
        else:
            raise TypeError("Your comparison directory is not valid >:(")

        if plot_option not in ["presentation", "maximize_graph"]:
            raise ValueError("Your plot option is not available.  Please select 'presentation' or 'maximize_graph' >:(")
        else:
            self.plot_option = plot_option

        if isinstance(animate, bool):
            self.animate = animate
        else:
            raise TypeError("Your animation option is not recognized >:(")

        if isinstance(store_root, bool):
            self.store_root = store_root
        else:
            raise TypeError("Your ROOT histogram storage option is not a boolean >:(")

        if isinstance(store_npz, bool):
            self.store_npz = store_npz
        else:
            raise TypeError("Your NPZ storage option is not a boolean >:(")

        if isinstance(suppress_print, bool):
            self.suppress_print = suppress_print
        else:
            raise TypeError("Your print suppression option is not a boolean >:(")

        if isinstance(store_f_dist, bool):
            self.store_f_dist = store_f_dist
        else:
            raise TypeError("Your frequency distribution storage option is not a boolean >:(")

        # if isinstance(optimize_t_offset, bool):                                             # ADDED #
        #     self.optimize_t_offset = optimize_t_offset                                      # ADDED #
        # else:                                                                               # ADDED #
        #     raise TypeError("Your offset time optimization option is not a boolean o_o")    # ADDED #

        if isinstance(multiplex_plots, str):
            self.multiplex_plots = [multiplex_plots]
        elif isinstance(multiplex_plots, list):
            self.multiplex_plots = multiplex_plots
        else:
            raise TypeError("Your multiplex plots option is not valid.")

    @_suppress_print
    def propagate(
        self,
        state_generator,
        ring, # the Ring object representing the storage ring and fields inside
        integration_method = "rk4",
        dt = 0.1, # the time-step in nanoseconds
        t_f = 1000, # ending time in ns
    ):
        if state_generator.plex == "simplex" and ring.mode == "simplex":
            subpropagator = _SPropagator(self.output, self.plot, self.display_options_s, self.display_options_m, self.p_acceptance_dir, self.plot_option, self.animate, self.store_root, self.store_npz, self.suppress_print, self.store_f_dist)
            print(f"Propagator initialized, working in /{self.output}.  Proceeding with simplex simulation.")
        else:
            subpropagator = _MPropagator(self.output, self.plot, self.p_acceptance_dir, self.plot_option, self.store_root, self.store_npz, self.suppress_print, self.store_f_dist, self.multiplex_plots)
            print(f"Propagator initialized, working in /{self.output}.  Proceeding with multiplex simulation.")
        lost = subpropagator.propagate(state_generator, ring, integration_method, dt, t_f)
        return lost

    @_suppress_print
    def align(self, state_generator, ring, integration_method = "rk4",
        dt = 0.1, # the time-step in nanoseconds
        t_f = 1000, # ending time in ns
        t_0 = 0,
        coarseRange = 100,
        coarseStep = 10,
        fineRange = 5,
        fineStep = 0.5
    ):
        if state_generator.plex == "multiplex" or ring.mode == "multiplex":
            raise ValueError("Your StateGenerator or Ring object is in multiplex mode!")
        else:
            aligner = _Aligner(self.output, state_generator, ring, integration_method)
            aligner.align(dt, t_f, t_0, coarseRange, coarseStep, fineRange, fineStep)

class _SPropagator:
    def __init__(
        self,
        output, # directory for the output plots and .npz files
        plot,
        display_options_s,
        display_options_m,
        p_acceptance_dir,
        plot_option,
        animate,
        store_root,
        store_npz,
        suppress_print,
        store_f_dist
    ):
        self.output = output
        self.plot = plot
        self.display_options_s = display_options_s
        self.display_options_m = display_options_m
        self.p_acceptance_dir = p_acceptance_dir
        self.plot_option = plot_option
        self.animate = animate
        self.store_root = store_root
        self.store_npz = store_npz
        self.suppress_print = suppress_print
        self.store_f_dist = store_f_dist
    
    def propagate(self, state_generator, ring, integration_method, dt, t_f):
        self.state_generator = state_generator
        self.ring = ring
        if integration_method in ("rk4", "rkn"):
            self.integration_method = integration_method
        else:
            raise ValueError(f"Your integration method '{integration_method}' is not recognized.")
        self.dt = dt
        self.t_f = t_f
        self.t_align = state_generator.zero

        # Generate the initial conditions from the given StateGenerator object and its parameters.
        init_state = self.state_generator.generate(self.integration_method, ring.n)

        print("Propagation begun.")
        begin_time = time.time()
        integrator = Integrator(integration_method = self.integration_method)
        results = integrator.integrate(init_states = init_state, ring = self.ring, dt = self.dt, t_f = self.t_f)
        self._print_results(begin_time, results)
        self._prep()
        _SPropagator._wipe(self.full_path, self.dir)
        self._store_results(results, init_state, ring.n)
        if self.store_f_dist:
            self._f_dist(results, ring.n)
        self._store_parameters(dt, t_f, ring)

        if not self.plot:
            print("No plots will be produced.")
        else:
            self._generate_plots()

    def _print_results(self, begin_time, results):
        self.sim_time = time.time() - begin_time
        minutes, seconds = int(self.sim_time // 60), self.sim_time % 60
        # Print results for single muon
        if self.state_generator.muons == 1:
            states, lost = results
            if lost:
                print(f"\nPropagation completed in {minutes} minutes, {seconds:.1f} seconds. Your muon is sad :(")
            else:
                print(f"\nPropagation completed in {minutes} minutes, {seconds:.1f} seconds. Your muon is happy! :)")

            print("FINAL STATE:")
            if self.integration_method == ("rk4" or "rkn"):
                print(f"x = {(states[-1][0] - aux.r_magic) * 1000} mm")
                print(f"\u03C6 = {states[-1][1] / np.pi * 180}\u00B0")
                print(f"v_r = {states[-1][2] * 1E-6} mm/ns")
                print(f"v_\u03C6 = {states[-1][3] / np.pi * 180 / 1E9} deg/ns")
                print(f"time = {states[-1][4] * 10**9} ns")
            elif self.integration_method == "optical":
                raise ValueError("this isnt done yet")

            self.lost_percentage = 0
        # Print results of multiple muons.
        else:
            self.lost_count = np.count_nonzero(results[1] == True)
            self.lost_percentage = self.lost_count / len(results[0]) * 100
            print(f"\nPropagation completed in {minutes} min, {seconds:.1f} s. {self.lost_count} out of {len(results[0])} (approx. {self.lost_percentage:.1f}%) muons are sad :(") #\u2248

    def _store_results(self, results, init_state, n):
        # Store results for a single muon
        muon_number = self.state_generator.muons
        if muon_number == 1:
            states, lost = results

            if self.integration_method == ("rk4" or "rkn") and self.store_npz:
                np.savez(self.full_path + "/states.npz", r = states[:,0], phi = states[:,1], vr = states[:,2], vphi = states[:,3], t = states[:,4])
            elif self.integration_method == "optical" and self.store_npz:
                raise ValueError("this isn't done yet")
            print("States saved.  ", end = "")

            if not lost:
                T, rco, A = self._calc_stats(states, muon_number)
                with open(f"{self.full_path}/stats.txt", "a") as file:
                    file.write("{:>10s} {:>10s} {:>10s}\n".format("T", "A", "r_co"))
                    file.write(f"{T:10.3f} {A:10.6f} {rco:10.5f}\n")
                print("Stats saved.", end = "")
            print("")    

        # Store results for multiple muons
        else:
            final_states, lost = results
            if self.store_root:
                self._root_hist2d(init_state, final_states, lost, n)

            # Saving all data to .npz files.
            if self.integration_method == ("rk4" or "rkn") and self.store_npz:
                np.savez(f"{self.full_path}/init_states.npz", r = init_state[:,0], phi = init_state[:,1], vr = init_state[:,2], vphi = init_state[:,3], t = init_state[:,4])
                np.savez(f"{self.full_path}/final_states.npz", r = final_states[:,0], phi = final_states[:,1], vr = final_states[:,2], vphi = final_states[:,3], t = final_states[:,4], lost = lost)
            elif self.integration_method == "optical" and self.store_npz:
                raise ValueError("this isnt done yet")
            print("States saved.  ", end = "")
            
            if self.lost_count != len(final_states):
                avg_x, sigma_x, C_E = self._calc_stats(results, muon_number)
                with open(f"{self.full_path}/stats.txt", "a") as file:
                    file.write("{:>10s} {:>10s} {:>10s}\n".format("avg_x", "sigma_x", "C_E"))
                    file.write(f"{avg_x:10.3f} {sigma_x:10.6f} {C_E:10.5f}\n")
                print("Stats saved.", end = "")
            print("")

    def _store_parameters(self, dt, t_f, ring):
        # Save the general details of the simulation results into a parameters file.
        with open(self.full_path + "/params_readable.txt", "w") as file:
            file.write("SIMULATION PARAMETERS:\n")
            if self.integration_method == "rk4":
                file.write(f"Integrator: 4th order Runge-Kutta method\n")
            elif self.integration_method == "rkn":
                file.write(f"Integrator: 4th order Runge-Kutta-Nyström method\n")
            elif self.integration_method == "optical":
                file.write(f"Integrator: Transfer matrix method\n")
            file.write(f"Time step: {dt} ns\n")
            file.write(f"Simulation end time: {t_f} ns\n")
            file.write(f"Real-time simulation length: {self.sim_time} sec\n")
            try:
                file.write(f"Lost muons: {self.lost_count}\n\n")
            except AttributeError:
                file.write("\n")

        sim_params = {'int_method': self.integration_method, 'dt': dt, 't_f': t_f}

        df = pd.DataFrame(sim_params, index = [0])

        try:
            with pd.ExcelWriter(self.full_path + "/parameters.xlsx", engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = 'sim', index = False)
        except FileNotFoundError:
            with pd.ExcelWriter(self.full_path + "/parameters.xlsx", mode = 'w') as writer:
                df.to_excel(writer, sheet_name = 'sim', index = False)

        ring._store_parameters(self.full_path)
        self.state_generator._store_parameters(self.full_path)

    def _generate_plots(self):
        if self.lost_percentage == 100:
            print("All muons lost, no plots created.")
        else:
            plotter = Plotter(data_directory = self.dir, output_directory = self.dir, p_acceptance_data = self.p_acceptance_dir, t_align = self.t_align, pdf_name = "plots", display_options = self.display_options, plot_option = self.plot_option, animate = self.animate)
            plotter.plot()

    def _root_hist2d(self, init_state, final_states, lost, n):
        # Save the momentum / injection time correlation plots as ROOT histogram if turned on.
        if self.integration_method == ("rk4" or "rkn"):
            r, vr, vphi = final_states[:,0], final_states[:,2], final_states[:,3]
            p = aux.state_to_mom_cyl(r = r, vr = vr, vphi = vphi)
        else:
            raise ValueError("this isnt done yet")
        f = aux.p_to_f(momentum = p, n = n)
        inj = init_state[:,4] * 10**9

        f_s, inj_s = Plotter._mask_lost(lost, f, inj)

        heights, x_edges, y_edges = np.histogram2d(x = inj_s, y = f_s, bins = (100,100))

        xRescale = 1
        yRescale = 1
        rootFile = ROOT.TFile(f"{self.full_path}/momentum_time.root", "RECREATE")
        histogram = ROOT.TH2F(
            "joint",
            ";Injection Time (ns);Cyclotron Frequency (kHz)",
            len(x_edges) - 1,
            x_edges[0] * xRescale,
            x_edges[-1] * xRescale,
            len(y_edges) - 1,
            y_edges[0] * yRescale,
            y_edges[-1] * yRescale
        )
        rnp.array2hist(heights, histogram)
        histogram.ResetStats()
        histogram.Write()
        rootFile.Close()

    def _f_dist(self, results, n, width = 150, df = 2):
        edges = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)
        final_states, lost = results

        f = aux.p_to_f(aux.state_to_mom_cyl(r = final_states[:,0], vr = final_states[:,2], vphi = final_states[:,3]), n = n)
        f_s = Plotter._mask_lost(lost, f)
        h, edges = np.histogram(f_s, bins = edges)
        errors = np.sqrt(h) / (len(f_s) * df)
        h_normed = h / (len(f_s) * df)
        assert sum(h) == len(f_s)
        cov = np.diag(np.square(errors))
        np.savez(f"{self.full_path}/f_dist.npz", heights = h_normed, cov = cov)
    
    def _calc_stats(self, results, muon_number):
        if muon_number == 1:
            # T, rco, A
            lower_bound = int(-174 / self.dt)
            r, t = results[:,0], results[:,4]
            x = (r - 7.112) * 1000
            t *= 10**9
            r_max_idx = lower_bound + len(x) + np.argmax(x[lower_bound:])
            r_max_idx2 = r_max_idx + lower_bound + np.argmax(x[r_max_idx + lower_bound:r_max_idx])

            T = t[r_max_idx] - t[r_max_idx2]
            rco = (x[r_max_idx] + np.min(x[r_max_idx2:r_max_idx])) / 2
            A = (x[r_max_idx] - np.min(x[r_max_idx2:r_max_idx])) / 2
            return T, rco, A
            
        else:
            #avg_x, sigma_x, C_E
            final_states, lost = results
            r = final_states[:,0]
            phi = final_states[:,1]
            vr = final_states[:,2]
            vphi = final_states[:,3]
            t = final_states[:,4]
            rco = aux.p_to_rco(aux.state_to_mom_cyl(final_states[:,0], final_states[:,2], final_states[:,3]), self.ring.n)
            rco_s = Plotter._mask_lost(lost, rco)
            return np.mean(rco_s), np.std(rco_s, ddof = 1), aux.x_to_C_E(np.mean(rco_s), np.std(rco_s, ddof = 1), self.ring.n)

    def _prep(self):
        prefix = ""

        if self.state_generator.muons == 1:
            prefix = prefix + "1"
            self.display_options = self.display_options_s
        else:
            prefix = prefix + "2"
            self.display_options = self.display_options_m

        if self.state_generator.plex == "simplex":
            prefix = "s" + prefix + "-"
        else:
            prefix = "m" + prefix + "-"
            self.display_options = []

        self.dir = prefix + self.output

        self.full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + self.dir

    @staticmethod
    def _wipe(path, dirname):
        """Check if the directory exists, and if so, overwrite it."""
        if pathlib.Path(path).is_dir():
            print(f"Overwriting directory /{dirname}...")
            shutil.rmtree(path)
        else:
            print(f"Creating directory /{dirname}...")
        os.mkdir(path)

class _MPropagator:
    def __init__(
        self,
        output, # directory for the output plots and .npz files
        plot,
        p_acceptance_dir,
        plot_option,
        store_root,
        store_npz,
        suppress_print,
        store_f_dist,
        multiplex_plots
    ):
        self.output = output
        self.plot = plot
        if output is None:
            warnings.warn(message = "Your output is set to None, so no plots will be produced!", category = UserWarning, stacklevel = 2)
        self.p_acceptance_dir = p_acceptance_dir
        self.plot_option = plot_option
        self.store_root = store_root
        self.store_npz = store_npz
        self.suppress_print = suppress_print
        self.multiplex_plots = multiplex_plots
        self.store_f_dist = store_f_dist

    def propagate(self, state_generator, ring, integration_method, dt, t_f):
        self.dt = dt
        self.t_f = t_f

        if integration_method in ("rk4", "rkn", "optical"):
            self.integration_method = integration_method
        else:
            raise ValueError(f"Your integration method '{integration_method}' is not recognized.")
        
        self.state_generator = state_generator
        self.t_align = state_generator.zero

        if self.state_generator.plex == "multiplex":
            init_states = self.state_generator.generate(self.integration_method, ring.n)
            self.subruns = self.state_generator.loops
            rings = [ring] * self.subruns
        else:
            rings = _MRing(ring).generate_list()
            self.subruns, _ = ring.search()
            init_states = np.array([self.state_generator.generate(self.integration_method, ring.n, verbose = False) for ring in rings])     # For multiple rings, the init states should be identical between each run (i think) but I'm not sure that's what's implmented here #
        
        self._prep()
        self.plottable_subruns = []
        integrator = Integrator(integration_method = self.integration_method)
        print("Propagation begun.")
        run_begin = time.time()
        for i in range(self.subruns):
            self.current_subrun = i+1
            subrun_begin = time.time()
            self.ring = rings[i]
            results = integrator.integrate(init_states = init_states[i], ring = rings[i], dt = self.dt, t_f = self.t_f)
            subrun_length = time.time() - subrun_begin
            minutes, seconds = int(subrun_length // 60), subrun_length % 60
            print(f"Subrun [{self.current_subrun}/{self.subruns}] complete. Length: {minutes} min, {seconds:.1f} s. ", end = "")
            self._store_results(results, init_states[i], self.ring.n, f"{self.tmp_path}/subrun_{self.current_subrun}_{self.subruns}")       # tony chanaged ring.n to self.ring.n
            if self.store_f_dist:
                self._f_dist(f"{self.tmp_path}/subrun_{self.current_subrun}_{self.subruns}", results, ring.n)

        self.sim_length = time.time() - run_begin
        minutes, seconds = int(self.sim_length // 60), self.sim_length % 60
        print(f"Multiplex simulation complete. Total time: {minutes} min, {seconds:.1f} sec.s")
        
        if not self.plot:
            print("No output graphs/animations/results saved or produced.")
        else:
            self._store_parameters(dt, t_f, ring)
            if len(self.plottable_subruns) == 0:
                print("No subruns are available to plot - no plots produced.")
            else:
                self._generate_plots()
        _MPropagator._rename(self.tmp_path, self.full_path, self.dir)

    def _store_results(self, results, init_state, n, subrun_path):
        # Store results for a single muon
        if self.state_generator.muons == 1:
            states, lost = results

            if self.integration_method == ("rk4" or "rkn") and self.store_npz:
                np.savez(f"{subrun_path}/states.npz", r = states[:,0], phi = states[:,1], vr = states[:,2], vphi = states[:,3], t = states[:,4])
            elif self.integration_method == "optical" and self.store_npz:
                raise ValueError("this isn't done yet")
            print("States saved.  ", end = "")
            
            self.plottable_subruns.append(self.current_subrun)
            
            if not lost:
                T, rco, A = self._calc_stats(results)
                with open(f"{self.tmp_path}/stats.txt", "a") as file:
                    file.write(f"{self.current_subrun:>6d} {T:10.3f} {A:10.6f} {rco:10.5f} {lost}\n")
            else:
                T, rco, A = "n/a", "n/a", "n/a"
                with open(f"{self.tmp_path}/stats.txt", "a") as file:
                    file.write(f"{self.current_subrun:>6d} {T:10s} {A:10s} {rco:10s} {lost}\n")
            print("Stats saved.", end = "")
            print("")

        # Store results for multiple muons
        else:
            final_states, lost = results
            if self.store_root:
                self._root_hist2d(init_state, final_states, lost, n, subrun_path)
            # Saving all data to .npz files.
            if self.integration_method == ("rk4" or "rkn"):
                np.savez(f"{subrun_path}/init_states.npz", r = init_state[:,0], phi = init_state[:,1], vr = init_state[:,2], vphi = init_state[:,3], t = init_state[:,4])
                np.savez(f"{subrun_path}/final_states.npz", r = final_states[:,0], phi = final_states[:,1], vr = final_states[:,2], vphi = final_states[:,3], t = final_states[:,4], lost = lost)
            elif self.integration_method == "optical":
                raise ValueError("this isnt done yet")
            print("States saved.  ", end = "")

            if len(lost) != sum(lost):
                self.plottable_subruns.append(self.current_subrun)
                avg_x, sigma_x, C_E = self._calc_stats(results)
                with open(f"{self.tmp_path}/stats.txt", "a") as file:
                    file.write(f"{self.current_subrun:>6d} {avg_x:10.3f} {sigma_x:10.6f} {C_E:10.5f}  {( (1-(sum(lost) / len(lost))) * 100):>8.3f}\n")
                print("Stats saved.", end = "")
            print("")

    def _f_dist(self, subrun_path, results, n, width = 150, df = 2):
        edges = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)
        final_states, lost = results

        f = aux.p_to_f(aux.state_to_mom_cyl(r = final_states[:,0], vr = final_states[:,2], vphi = final_states[:,3]), n = n)
        f_s = Plotter._mask_lost(lost, f)
        h, edges = np.histogram(f_s, bins = edges)
        errors = np.sqrt(h) / (len(f_s) * df)
        h_normed = h / (len(f_s) * df)
        assert sum(h) == len(f_s)
        cov = np.diag(np.square(errors))
        np.savez(f"{subrun_path}/f_dist.npz", heights = h_normed, cov = cov)
    
    def _calc_stats(self, results):
        if self.state_generator.muons == 1:
            # T, rco, A
            states, lost = results
            lower_bound = int(-174 / self.dt)
            r, t = states[:,0], states[:,4]
            x = (r - 7.112) * 1000
            t *= 10**9
            r_max_idx = lower_bound + len(x) + np.argmax(x[lower_bound:])
            r_max_idx2 = r_max_idx + lower_bound + np.argmax(x[r_max_idx + lower_bound:r_max_idx])

            T = t[r_max_idx] - t[r_max_idx2]
            rco = (x[r_max_idx] + np.min(x[r_max_idx2:r_max_idx])) / 2
            A = (x[r_max_idx] - np.min(x[r_max_idx2:r_max_idx])) / 2
            return T, rco, A
            
        else:
            #avg_x, sigma_x, C_E
            final_states, lost = results
            rco = aux.p_to_rco(aux.state_to_mom_cyl(final_states[:,0], final_states[:,2], final_states[:,3]), self.ring.n)
            rco_s = Plotter._mask_lost(lost, rco)
            return np.mean(rco_s),  np.std(rco_s, ddof = 1), aux.x_to_C_E(np.mean(rco_s), np.std(rco_s, ddof = 1), self.ring.n)

    def _store_parameters(self, dt, t_f, ring):
        # Save the general details of the simulation results into a parameters file.
        with open(self.tmp_path + "/params_readable.txt", "w") as file:
            file.write("SIMULATION PARAMETERS:\n")
            if self.integration_method == "rk4":
                file.write(f"Integrator: 4th order Runge-Kutta method\n")
            if self.integration_method == "rkn":
                file.write(f"Integrator: 4th order Runge-Kutta-Nyström method \n")
            elif self.integration_method == "optical":
                file.write(f"Integrator: Transfer matrix method\n")
            file.write(f"Time step: {dt} ns\n")
            file.write(f"Simulation end time: {t_f} ns\n")
            file.write(f"Real-time simulation length: {self.sim_length} sec\n")

        sim_params = {'int_method': self.integration_method, 'dt': dt, 't_f': t_f}

        df = pd.DataFrame(sim_params, index = [0])

        try:
            with pd.ExcelWriter(self.tmp_path + "/parameters.xlsx", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = 'sim', index = False)
        except FileNotFoundError:
            with pd.ExcelWriter(self.tmp_path + "/parameters.xlsx", mode = 'w') as writer:
                df.to_excel(writer, sheet_name = 'sim', index = False)

        ring._store_parameters(self.tmp_path)
        self.state_generator._store_parameters(self.tmp_path)

    def _generate_plots(self):
        plotter = Plotter(data_directory = self.tmp_dir, output_directory = self.tmp_dir, p_acceptance_data = self.p_acceptance_dir, t_align = self.t_align, plot_option = self.plot_option, multiplex_plots = self.multiplex_plots)
        plotter.plottable_subruns = self.plottable_subruns
        plotter.plot()

    def _root_hist2d(self, init_state, final_states, lost, n, subrun_path):
        # Save the momentum / injection time correlation plots as ROOT histogram if turned on.
        if self.integration_method == ("rk4" or "rkn"):
            r, vr, vphi = final_states[:,0], final_states[:,2], final_states[:,3]
            p = aux.state_to_mom_cyl(r = r, vr = vr, vphi = vphi)
        else:
            raise ValueError("this isnt done yet")
        f = aux.p_to_f(momentum = p, n = n)
        inj = init_state[:,4] * 10**9

        f_s, inj_s = Plotter._mask_lost(lost, f, inj)

        heights, x_edges, y_edges = np.histogram2d(x = inj_s, y = f_s, bins = (100,100))

        xRescale = 1
        yRescale = 1
        rootFile = ROOT.TFile(f"{subrun_path}/momentum_time.root", "RECREATE")
        histogram = ROOT.TH2F(
            "joint",
            ";Injection Time (ns);Cyclotron Frequency (kHz)",
            len(x_edges) - 1,
            x_edges[0] * xRescale,
            x_edges[-1] * xRescale,
            len(y_edges) - 1,
            y_edges[0] * yRescale,
            y_edges[-1] * yRescale
        )
        rnp.array2hist(heights, histogram)
        histogram.ResetStats()
        histogram.Write()
        rootFile.Close()

    def _prep(self):
        if self.state_generator.muons == 1:
            prefix = "m1-"
        else:
            prefix = "m2-"

        self.dir = prefix + self.output
        self.tmp_dir = "&TMP" + self.dir

        self.full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + self.dir
        self.tmp_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + self.tmp_dir
        if pathlib.Path(self.tmp_path).is_dir():
            print("The temporary directory already exists. ", end = "")
            while True:
                clear = input("Clear? [y/n] ")
                if clear in ["y", "Y", "N", "n"]:
                    break
                else:
                    print("Invalid clear option. ", end = "")
            if clear in ["y", "Y"]:
                print("Clearing existing temporary directory... ", end = "")
                shutil.rmtree(self.tmp_path)
                print("Cleared.")
            else:
                raise KeyboardInterrupt("Temporary directory not cleared, simulation ending.")
        os.mkdir(self.tmp_path)
        for i in range(self.subruns):
            os.mkdir(f"{self.tmp_path}/subrun_{i+1}_{self.subruns}")

        if self.state_generator.muons == 1:
            with open(f"{self.tmp_path}/stats.txt", "w") as file:
                file.write("{:>6s} {:>10s} {:>10s} {:>10s} {:}\n".format("Subrun", "T", "A", "r_co", "lost?"))
        else:
            with open(f"{self.tmp_path}/stats.txt", "w") as file:
                file.write("{:>6s} {:>10s} {:>10s} {:>10s} {:>9s}\n".format("Subrun", "avg_x", "sigma_x", "C_E", "store %"))
            
    @staticmethod
    def _rename(src_path, dst_path, dirname):
        """Check if the directory exists, and if so, overwrite it."""
        if pathlib.Path(dst_path).is_dir():
            print(f"Overwriting directory /{dirname}...")
            shutil.rmtree(dst_path)
        else:
            print(f"Creating directory /{dirname}...")
        os.rename(src_path, dst_path)


class _Aligner:
    def __init__(self, output, sg, rg, integration_method):
        self.dir = 'a-' + output
        self.full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + self.dir
        if pathlib.Path(self.full_path).is_dir():
            print(f"Overwriting '/{self.dir}'...")
            shutil.rmtree(self.full_path)
        else:
            print(f"Creating '/{self.dir}'...")
        os.mkdir(self.full_path)
        self.generator = sg
        self.rg = rg
        self.integration_method = integration_method
        if self.generator.muons < 5E4:
            print("Warning: results may be sus due to low statistics.")

    def align(self, dt, t_f, t_0, coarseRange, coarseStep, fineRange, fineStep):
        self.dt = dt
        self.t_f = t_f
        self.t_0 = t_0
        self.coarseRange = coarseRange
        self.coarseStep = coarseStep
        self.fineRange = fineRange
        self.fineStep = fineStep
        print(f"Beginning alignment of injection pulse '{self.generator.t.dir[0]}.dat' and kicker file '{self.rg.b_k.B}.dat'.")
        self._store_auxdata()

        coarse_t0 = self.coarse_search(self.t_0, self.coarseRange, self.coarseStep) 
        fine_t0 = self.fine_search(coarse_t0, self.fineRange, self.fineStep)
        final_t0, final_store = self.fit()
        print(f"Final t0: {final_t0: 7.2f} ns, rate: {final_store:5.3f} %")
        self._plot()

    def coarse_search(self, t_0, coarseRange, coarseStep):

        total_subruns = int(((coarseRange / coarseStep) * 2 ) + 1)
        self.coarse_t0s = t_0 + np.linspace(-1 * coarseRange, coarseRange, num = total_subruns, endpoint = True)
        self.coarse_rates = np.zeros(shape = total_subruns)
    
        sg = StateGenerator(mode = self.generator.mode, seed = self.generator.seed, muons = self.generator.muons, initial_offset = self.generator.initial_offset, momentum = self.generator.momentum, offset = self.generator.offset, f = self.generator.f, alpha = self.generator.alpha, phi_0 = self.generator.phi_0,
            t = dist.Custom(dir = self.generator.t.dir, zero = self.coarse_t0s))            # will be in multiplex mode
        init_states = sg.generate(self.integration_method, self.rg.n, verbose = False)      # shape is (total_subruns) x (muons) x (5) #
        integrator = Integrator(integration_method = self.integration_method)
        print("Coarse scanning...", end = '')
        search_begin = time.time()
        for i in range(total_subruns):
            subrun_begin = time.time()
            final_states, lost = integrator.integrate(init_states = init_states[i], ring = self.rg, dt = self.dt, t_f = self.t_f)
            subrun_length = time.time() - subrun_begin
            minutes, seconds = int(subrun_length // 60), subrun_length % 60
            if minutes == 0:
                print(f"\rCoarse scanning... Subrun [{i+1}/{total_subruns}] complete. Length: {seconds:.1f} s.    ", end = "")
            else:
                print(f"\rCoarse scanning... Subrun [{i+1}/{total_subruns}] complete. Length: {minutes} min, {seconds:.1f} s. ", end = "")
            self.coarse_rates[i] = (1-(sum(lost) / len(lost))) * 100
        
        idx = np.argmax(self.coarse_rates)
        coarse_t0 = self.coarse_t0s[idx]
        if idx in [0, total_subruns - 1]:
            yn = input(f"Coarse scan: suitable minimum not found within given range; re-search with range [{coarse_t0 - coarseRange}, {coarse_t0 + coarseRange}]? [y/n] ")
            if yn in ['n', 'N', 0]:
                print("Coarse scan failed, exiting.")
                sys.exit()
            elif yn in ["Y", 'y', 1]:
                print("Restarting coarse scan with new range.")
                coarse_t0 = self.coarse_search(t_0 = coarse_t0, coarseRange = coarseRange, coarseStep = coarseStep)
        else:
            print(f"\nCoarse scan successful in {time.time() - search_begin:5.3f} s: found t0 = {coarse_t0}, rate = {self.coarse_rates[idx]:4.2f}, subrun {idx+1}/{total_subruns}")
        self._store_scan(self.coarse_t0s, self.coarse_rates, "coarse")
        return coarse_t0

    def fine_search(self, t_0, fineRange, fineStep):

        total_subruns = int(((fineRange / fineStep) * 2 ) + 1)
        self.fine_t0s = t_0 + np.linspace(-1 * fineRange, fineRange, num = total_subruns, endpoint = True)
        self.fine_rates = np.zeros(shape = total_subruns)
    
        sg = StateGenerator(mode = self.generator.mode, seed = self.generator.seed, muons = self.generator.muons, initial_offset = self.generator.initial_offset, momentum = self.generator.momentum, offset = self.generator.offset, f = self.generator.f, alpha = self.generator.alpha, phi_0 = self.generator.phi_0,
            t = dist.Custom(dir = self.generator.t.dir, zero = self.fine_t0s))
        init_states = sg.generate(self.integration_method, self.rg.n, verbose = False)
        integrator = Integrator(integration_method = self.integration_method)
        print("Fine scanning...", end = '')
        search_begin = time.time()
        
        for i in range(total_subruns):
            subrun_begin = time.time()
            final_states, lost = integrator.integrate(init_states = init_states[i], ring = self.rg, dt = self.dt, t_f = self.t_f)
            subrun_length = time.time() - subrun_begin
            minutes, seconds = int(subrun_length // 60), subrun_length % 60
            if minutes == 0:
                print(f"\rFine scanning... Subrun [{i+1}/{total_subruns}] complete. Length: {seconds:.1f} s.                ", end = "")
            else:
                print(f"\rFine scanning... Subrun [{i+1}/{total_subruns}] complete. Length: {minutes} min, {seconds:.1f} s. ", end = "")
            self.fine_rates[i] = (1-(sum(lost) / len(lost))) * 100
        
        idx = np.argmax(self.fine_rates)
        fine_t0 = self.fine_t0s[idx]
        if idx in [0, total_subruns - 1]:
            yn = input(f"\nFine scan: suitable minimum not found within given range; re-search with range [{fine_t0 - fineRange}, {fine_t0 + fineRange}]? [y/n] ")
            if yn in ['n', 'N', 0]:
                print("Fine scan failed, exiting.")
                sys.exit()
            elif yn in ["Y", 'y', 1]:
                print("Restarting fine scan with new range.")
                fine_t0 = self.fine_search(t_0 = fine_t0, fineRange = fineRange, fineStep = fineStep)
        else:
            print(f"\nFine scan successful in {time.time() - search_begin:5.3f} s: found t0 = {fine_t0}, rate = {self.fine_rates[idx]:4.2f}, subrun {idx+1}/{total_subruns}")
        self._store_scan(self.fine_t0s, self.fine_rates, "fine")
        return fine_t0

    def fit(self):
        idx = np.argmax(self.fine_rates)
        a, b, c = np.polyfit(self.fine_t0s[idx-1:idx+2], self.fine_rates[idx-1:idx+2], deg = 2)
        final_t0, final_store = -b / (2 * a), c - (b**2 / 4 / a)
        return final_t0, final_store

    def _store_scan(self, t0s, rates, filename):
        with open(f"{self.full_path}/{filename}.txt", "w") as file:
            file.write("{:>6s} {:>10s} {:>10s}\n".format("Subrun", "t0", "store %"))
            for idx, t0 in enumerate(t0s):
                file.write(f"{idx+1:>6d} {t0:10.5f} {rates[idx]:10.5f}\n")

    def _store_auxdata(self):
        with open(f"{self.full_path}/aux.txt", "w") as file:
            file.write(f"{self.generator.t.dir[0]}\n{self.rg.b_k.B}, {self.rg.b_k.b_norm}, {self.rg.b_k.t_norm}")
    
    def _plot(self):
        plotter = AlignPlotter(self.dir)
        plotter.plot()
        print('Plots finished.')