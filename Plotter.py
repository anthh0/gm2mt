import warnings
import pathlib
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as ani
import PyPDF4 as pypdf
import scipy.optimize as opt

import gm2mt.auxiliary as aux
import gm2mt.Distributions as dist
from gm2mt.StateGenerator import StateGenerator
from gm2mt.Ring import Ring, _MRing

class Plotter:
    def __init__(
        self,
        data_directory,
        output_directory,
        p_acceptance_data = None,
        t_align = 0,
        pdf_name = "plots",
        display_options = [],
        plot_option = "presentation",
        animate = False,
        multiplex_plots = []
    ):
        if isinstance(data_directory, str):
            self.data_dir = data_directory
            self.data_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + self.data_dir
            if not pathlib.Path(self.data_path).is_dir():
                raise ValueError("Your given data directory does not exist.")
        else:
            raise ValueError("Your data directory is not valid.")
        
        if isinstance(output_directory, str):
            self.output_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + output_directory
            if not pathlib.Path(self.output_path).is_dir():
                os.mkdir(self.output_path)
        else:
            raise ValueError("Your output directory is not valid.")

        if isinstance(display_options, list):
            self.display_options = display_options
        elif isinstance(display_options, str):
            self.display_options - [display_options]
        else:
            raise ValueError("Your display options are not a list or string.")
        
        if plot_option not in ["presentation", "maximize_graph"]:
            raise ValueError("Your plot option is not available.  Please select 'presentation' or 'maximize_graph' >:(")
        else:
            self.plot_option = plot_option

        if isinstance(animate, bool):
            self.animate = animate
        else:
            raise ValueError("Your animation option is not valid.")

        if isinstance(pdf_name, str):
            self.pdf_name = pdf_name
        else:
            raise ValueError("Your PDF name is not a string.")

        if isinstance(p_acceptance_data, str):
            if p_acceptance_data == "":
                self._p_acc_dir = p_acceptance_data
            else:
                if pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + "/momentum_acceptance_data/" + p_acceptance_data).is_dir():
                    self._p_acc_dir = p_acceptance_data
                else:
                    warnings.warn(message = f"Your momentum acceptance data directory '{p_acceptance_data}' was not found and will be ignored.", category = UserWarning, stacklevel = 3)
                    self._p_acc_dir = None
        elif p_acceptance_data is None:
            self._p_acc_dir = p_acceptance_data
        else:
            raise TypeError("Your momentum acceptance data parameter is not a string or 'None'!")

        if isinstance(t_align, (int, float)):
            self.t_align = t_align
        else:
            raise TypeError(f"Your align time '{t_align}' is not a float")

        if isinstance(multiplex_plots, str):
            self.multiplex_plots = [multiplex_plots]
        elif isinstance(multiplex_plots, list):
            self.multiplex_plots = multiplex_plots

    def settings(self):
        if self.plot_option == "presentation":
            plt.rcParams["figure.figsize"] = (8,6)
            plt.rcParams["figure.titlesize"] = 9
            plt.rcParams["axes.titlesize"] = 17
            plt.rcParams["axes.labelsize"] = 12
        elif self.plot_option == "maximize_graph":
            plt.rcParams["figure.figsize"] = (21,10)
            plt.rcParams["figure.titlesize"] = 9
            plt.rcParams["axes.titlesize"] = 15
            plt.rcParams["axes.labelsize"] = 12
            plt.rcParams["font.size"] = 14
        
        # plt.rcParams["xtick.labelsize"] = 10
        # plt.rcParams["ytick.labelsize"] = 10
        
        plt.rcParams["font.family"] = "sans-serif"
        
        plt.rcParams["image.cmap"] = 'plasma'
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams["text.usetex"] = False
        plt.rcParams["axes.formatter.use_mathtext"] = True

        # Make axis tick marks face inward.
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        # Draw axis tick marks all around the edges.
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True

        # Draw minor axis tick marks in between major labels.
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True

        # Make all tick marks longer.
        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["xtick.minor.size"] = 4
        plt.rcParams["ytick.major.size"] = 8
        plt.rcParams["ytick.minor.size"] = 4
    
    def plot(self):
        self.ring = Ring.load(dir = self.data_dir)
        
        sg_mode = StateGenerator.load(dir = self.data_dir).plex
        rg_mode = self.ring.mode
        if sg_mode == "multiplex" or rg_mode == "multiplex":
            self._plotM()
        else:
            self._plotS()

    def _plotS(self):
        self.pdf_path = f"{self.output_path}/{self.pdf_name}.pdf"
        self.labels = []

        # Set the Matplotlib settings.
        self.settings()
        # import ring, identify integration method and rudder
        self._extract(self.data_path)

        # Create an empty PDF.
        plots_pdf = PdfPages(self.pdf_path)
        plots_pdf.close()
        
        if self.number == 1: # Single muon plots
            fig_trajectory = self._plot_trajectory()
            # Plot of the radial offset over time, with kicker highlighting.
            fig_offset = self._plot_offset()
            # Plot of the angular position over time, with kicker highlighting.
            fig_phi = self._plot_phi()
            # Plot of the radial speed over time
            fig_vr = self._plot_vr()
            #Plot of the angular speed over time.
            fig_vphi = self._plot_vphi()
            # Plot of the momentum over time.
            fig_p = self._plot_p()
            # Plot of the experienced kicker field
            fig_kicker = self._plot_kicker()
            # Plot of an animation showing the muon's path through the ring, rescaled somewhat for visibility
            if self.animate:
                fig_ani = self._animate()

            if "trajectory" not in self.display_options:
                plt.close(fig_trajectory)
            if "offset" not in self.display_options:
                plt.close(fig_offset)
            if "phi" not in self.display_options:
                plt.close(fig_phi)
            if "vr" not in self.display_options:
                plt.close(fig_vr)
            if "vphi" not in self.display_options:
                plt.close(fig_vphi)
            if "p" not in self.display_options:
                plt.close(fig_p)
            if "kicker" not in self.display_options:
                plt.close(fig_kicker)
            if "animation" not in self.display_options and self.animate:
                plt.close(fig_ani)

            print("Plots saved.")
            plt.show()

        elif self.number == 2: # Multimuon plots
            # Momentum distribution plots.
            fig_p = self._plot_p_dist()
            # Final frqeuency distribution.
            fig_f = self._plot_f_dist()
            # Momentum / injection time distribution plots.
            fig_mt3 = self._plot_mt3_dist()
            # Momentum / injection time distribution plots.
            fig_mt1 = self._plot_mt1_dist()
            # Injection pulse compared to kicker pulse
            fig_pulse = self._plot_pulse()
            # Spatial distributions of surviving muons at end of simulation
            fig_spatial = self._plot_spatial()
            # Information on the injection profiles of the incoming and outgoing muon beams.
            fig_inj = self._plot_inj()
            # 2D histograms of the initial position and angle.
            fig_x_alpha = self._plot_x_alpha()

            if "p" not in self.display_options:
                plt.close(fig_p)
            if "f" not in self.display_options:
                plt.close(fig_f)
            if "mt3" not in self.display_options:
                plt.close(fig_mt3)
            if "mt1" not in self.display_options:
                plt.close(fig_mt1)
            if "pulse" not in self.display_options:
                plt.close(fig_pulse)
            if "spatial" not in self.display_options:
                plt.close(fig_spatial)
            if "inj" not in self.display_options:
                plt.close(fig_inj)
            if "x_alpha" not in self.display_options:
                plt.close(fig_x_alpha)
            
            print("Plots saved.")
            plt.show()
    
    def _plotM(self):
        self.settings()
        ring = Ring.load(dir = self.data_dir)
        sg = StateGenerator.load(dir = self.data_dir)
        self.muons = sg.muons

        if sg.plex == "multiplex":
            subruns, self.labels = sg.search()
            rings = [ring] * subruns
        elif ring.mode == "multiplex":
            subruns, self.labels = self.ring.search()
            rings = _MRing(ring = ring).generate_list()
        else:
            raise RuntimeError("Neither ring nor state generator were in multiplex mode...")

        if 'plottable_subruns' not in vars(self).keys():
            print('Plottable subruns not given, searching... ', end = "\r")
            self._find_plottable_subruns(subruns)
            print(f'Plottable subruns not given, searching... Found: {self.plottable_subruns}     ')
        else:
            print('Plottable subruns already set.')
        
        if len(self.plottable_subruns) == 0:
            print("No plottable subruns - no plots produced.")
        else:
            self._prep()
            for plot in self.multiplex_plots:
                if plot == "stats":
                    continue
                self.pdf_path = f"{self.output_path}/{plot}_comp.pdf"
                plots_pdf = PdfPages(self.pdf_path)
                plots_pdf.close()
                for subrun in self.plottable_subruns:
                    self.ring = rings[subrun - 1]
                    self._extract(f"{self.data_path}/subrun_{subrun}_{subruns}")
                    self.current_label = self.labels[subrun - 1]
                    fig = self.plot_funcs[plot]()
                print(f"Plot {plot}_comp.pdf constructed.")

            if "stats" in self.multiplex_plots:
                self._plot_stats()
        
    #region SINGLE MUON GRAPH FUNCTIONS:
    def _plot_trajectory(self):
        # print(f"r: {self.r}, phi: {self.phi}")
        
        scaled_r_min = 10                                                               # scale the r values and set up the theta values
        scaled_r_max = 15
        circle = np.linspace(0, 2*np.pi, 360)
        scaled_r0 = np.full(shape = len(circle), fill_value = (scaled_r_min + scaled_r_max)/2)
        scaled_r = Plotter._rescale(self.r, self.ring.r_min, self.ring.r_max, scaled_r_min, scaled_r_max)
                                                                                        # create a polar Axes, which interprets all (x,y) pairs as (theta,r) pairs 
        fig, ax = plt.subplots(1, 1, subplot_kw = dict(projection = "polar"), constrained_layout = True)

        r_height = scaled_r_max - scaled_r_min                                          # highlight the ring, kicker, and quad regions using polar bar plots
        k_angle, k_width = (aux.k1_i + aux.k3_f)/2, (aux.k3_f - aux.k1_i)
        q_angles, q_widths = [(aux.quad_lims['Q' + str(i+1)][1] + aux.quad_lims['Q' + str(i+1)][0])/2 for i in range(4)], 4*[(aux.Q1L_f - aux.Q1S_i)]
        ax.bar(0, r_height, width = 2*np.pi, bottom = scaled_r_min, color = (0.4, 0.3, 1.0))
        ax.bar(k_angle, r_height, width = k_width, bottom = scaled_r_min, color = (0.6, 1.0, 1.0), label = "Kicker region")
        ax.bar(q_angles, r_height, width = q_widths, bottom = [scaled_r_min]*4, color = [(1.0, 1.0, 0.6)]*4, label = "Quad regions")

        ax.plot(circle, scaled_r0, linewidth = 0.3, color = 'k')                        # add ring boundaries n such
        for phi in np.linspace(0, 2*np.pi, 9):
            ax.plot([phi, phi], [scaled_r_min, scaled_r_max], linewidth = 0.1, color = 'k')
        ax.plot(circle, scaled_r_min*np.ones(shape = len(circle)), 'k')
        ax.plot(circle, scaled_r_max*np.ones(shape = len(circle)), 'k')

        one_rot_idx = np.argmax(np.extract([phi <= 2*np.pi for phi in self.phi], self.phi))
        ax.plot(self.phi[:one_rot_idx], scaled_r[:one_rot_idx], 'r', label = "Muon trajectory")       # plot the trajectory data

        ax.set_title("Muon Trajectory")                                                 # boring Axes configuration
        ax.spines['polar'].set_visible(False)
        ax.grid(visible = False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.legend(loc = "lower right")

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/trajectory.pdf")                          # add figure to the output pdf

        return fig

    def _plot_offset(self):
        r_max_offset = (self.ring.r_max - self.ring.r_min) / 1.5 * 1000                 # Divide by 1.5 instead of 2 for a little extra padding
        r_min_offset = -1 * r_max_offset

        fig, ax = plt.subplots()
        
        ax.plot(self.t, self.r_offset)
        ax.plot([self.t[0], self.t[-1]], [0,0], color = 'gray', linestyle = '--', dashes = (5, 5), linewidth = 0.7)

        ax.set_title("Radial Offset vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel('Radial Offset (mm)')
        ax.set_yticks(np.linspace(r_min_offset, r_max_offset, 7))
        self._kicker_regions(ax, self.phi, self.t)
        self._quad_regions(ax, self.phi, self.t)
        ax.legend()
        
        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/offset.pdf")

        return fig
        
    def _plot_phi(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.phi)

        ax.set_title("Angle vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(r'$\phi$ (radians)')
        self._kicker_regions(ax, self.phi, self.t)
        self._quad_regions(ax, self.phi, self.t)
        ax.legend()

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/phi.pdf")

        return fig

    def _plot_vr(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.vr)
        ax.set_title("Radial Speed vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel('Radial Speed (mm/ns)')
        self._kicker_regions(ax, self.phi, self.t)
        self._quad_regions(ax, self.phi, self.t)
        ax.ticklabel_format(useOffset = False)
        ax.legend()

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/vr.pdf")

        return fig

    def _plot_vphi(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.vphi)
        ax.set_title("Angular Speed vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel('Angular Speed (deg/ns)')
        self._kicker_regions(ax, self.phi, self.t)
        self._quad_regions(ax, self.phi, self.t)
        ax.ticklabel_format(useOffset = False)
        ax.legend()

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/vphi.pdf")

        return fig

    def _plot_p(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.p)
        ax.set_title("Momentum vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel('Momentum (GeV)')
        self._kicker_regions(ax, self.phi, self.t)
        self._quad_regions(ax, self.phi, self.t)
        ax.ticklabel_format(style = 'sci', useMathText=True)
        ax.legend()

        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/p.pdf")

        return fig

    def _plot_kicker(self):
        fig, ax = plt.subplots()

        kfield = self.ring.b_k.b_k * 10_000                                         # Kicker strength in Gauss
        kfield_t_list = self.ring.b_k.t_list * 1E9                                  # Kicker time in Seconds
        entrance_times, exit_times = Plotter._find_in_out_times(self.phi, aux.k1_i, aux.k3_f, self.t, self.ring.b_k.kick_max)
        if len(exit_times) != 0:
            time_spread = exit_times[-1] - entrance_times[0]                        # Modified the plot conditional to better constrain kicker plot
            indices = [idx for idx, val in enumerate(kfield_t_list) if val > entrance_times[0] - 0.4*time_spread and val < exit_times[-1] + 0.4*time_spread]
            ax.plot([kfield_t_list[idx] for idx in indices], [kfield[idx] for idx in indices])
        else:
            raise ValueError("Can't plot kicker regions since muon never entered a kicker region")
        midtimes = []
        for i in range(len(entrance_times)):
            midtimes.append((entrance_times[i] + exit_times[i])/2)

        kick_strengths = np.interp(midtimes, kfield_t_list, kfield)
        kick_strengths_enter = np.interp(entrance_times, kfield_t_list, kfield)
        kick_strengths_exit = np.interp(exit_times, kfield_t_list, kfield)
        # print(f"Mid kick times is/are {midtimes}, entrance {entrance_times} exit {exit_times}")
        # print(kick_strengths)
        # print(kick_strengths_enter)
        # print(kick_strengths_exit)
        ax.set_title("Kicker Strength vs. Time")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel('Kicker Field Strength (G)')
        ax.axvline(self.t[0], linestyle = "--", color = "gray", label = "injection time")
        self._kicker_regions(ax, self.phi, self.t)

        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/kicker.pdf")

        return fig

    def _animate(self):
        # Perform a rescaling on the radial data, ONLY for use in the animation
        scaled_r_min, scaled_r_max = 6, 9
        scaled_r_0 = (scaled_r_max + scaled_r_min) / 2
        scaled_r = Plotter._rescale(self.r, self.ring.r_min, self.ring.r_max, scaled_r_min, scaled_r_max)
        
        # Some auxiliary data for some helpful plot elements showing the ring's physical boundaries
        full_circle = np.linspace(0, 2*np.pi, 100)
        inner = np.full(len(full_circle), scaled_r_min)
        outer = np.full(len(full_circle), scaled_r_max)
        midradius = np.full(len(full_circle), scaled_r_0)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(full_circle, inner, color = 'gray', linewidth = 0.5)
        ax.plot(full_circle, outer, color = 'gray', linewidth = 0.5)
        ax.plot(full_circle, midradius, color = 'gray', linestyle = 'dotted', linewidth = 0.5)
        ax.spines['polar'].set_visible(False)
        path, = ax.plot(self.phi, scaled_r)
        time = ax.text(3, 3, s = "Time: 0 ns", bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.5))
        ax.fill_between(np.linspace(aux.k1_i, aux.k3_f, 100), scaled_r_min, scaled_r_max, color = 'red', alpha = 0.3, label = 'Kicker')
        ax.set_title("Muon path (scaled)")
        self._labels(fig)
        ax.set_rlim(bottom = 0, top = scaled_r_max)
        ax.set_rticks([])
        ax.tick_params(axis='both', which='major', labelsize=7)
        fig.tight_layout(pad = 1)
        angle = np.deg2rad(67.5)
        ax.legend(loc="lower left", bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

        def update(num, phi, scaled_r, path, time):
            path.set_data(phi[:num], scaled_r[:num])
            time.set_text("Time: " + str(round(self.t[num], 1)) + " ns")
            return path, time

        path_animation = ani.FuncAnimation(fig, update, len(self.phi), fargs = [self.phi, scaled_r, path, time], interval = 25, blit = True)
        writergif = ani.PillowWriter(fps = 90)
        path_animation.save(f"{self.output_path}/path.gif", writer = writergif)

        self._merge(fig, f"{self.output_path}/ani.pdf")

        return fig
    #endregion

    # region MULTIMUON GRAPH FUNCTIONS:
    def _plot_p_dist(self, bin_num = 250):
        (p_avg, p_std), (p_s_avg, p_s_std), (p_i_avg, p_i_std) = Plotter._calculate_stats(self.p, self.p_s, self.p_i)

        fig, axs = plt.subplots(1, 2, sharey = 'row')

        x_min = self.p_i.min()
        x_max = self.p_i.max()

        axs[0].hist(self.p_i, bins = bin_num)
        axs[0].set_title(f"Initial (bins = {bin_num})")
        axs[0].set_xlabel("Momentum (GeV)")
        axs[0].set_ylabel('Counts')
        # axs[0].set_xlim([x_min, x_max])
        axs[0].text(0.02, 0.97, s = f"$\mu_p = {p_i_avg:.3f}$ GeV\n$\sigma_p = {p_i_std:.5f}$ GeV", ha = 'left', va = 'top', ma = 'left', transform = axs[0].transAxes, bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.2))
        
        # axs[1].hist(self.p, bins = bin_num)
        # axs[1].set_title("Final")
        # axs[1].set_xlabel("Momentum (GeV)")
        # axs[1].text(0.02, 0.97, s = f"$\mu_p = {p_avg:.3f}$ GeV\n$\sigma_p = {p_std:.5f}$ GeV", ha = 'left', va = 'top', ma = 'left', transform = axs[1].transAxes, bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.2))
        
        axs[1].hist(self.p_s, bins = bin_num)
        axs[1].set_title("Surviving")
        axs[1].set_xlabel("Momentum (GeV)")
        # axs[1].set_xlim([x_min, x_max])
        axs[1].text(0.02, 0.97, s = f"$\mu_p = {p_s_avg:.3f}$ GeV\n$\sigma_p = {p_s_std:.5f}$ GeV\nstore rate \u2248 {self.store_percentage:.2f}%", ha = 'left', va = 'top', ma = 'left', transform = axs[1].transAxes, bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.2))

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/p_dist.pdf")

        return fig

    def _plot_f_dist(self):
        f_s_avg, f_s_std = Plotter._calculate_stats(self.f_s)
        f_s_heights, f_s_edges = np.histogram(self.f_s, bins = np.arange(6630, 6781, 1))
        f_s_centers = (f_s_edges[1:] + f_s_edges[:-1])/2

        fig, ax = plt.subplots()
        ax.set_title("Final frequency distribution")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Entries / kHz")
        ax.tick_params(which = 'both', direction = 'in')
        ax.plot(f_s_centers, f_s_heights, color = 'black')
        ax.grid(b = True, alpha = 0.25)
        ax.set_ylim(bottom = 0, top = None)
        ax.set_xlim((6630, 6780))
        ax.text(0.02, 0.97, s = f"$\mu_f = {f_s_avg: .1f}$ kHz\n$\sigma_f = {f_s_std: .2f}$ kHz", ha = 'left', va = 'top', ma = 'left', transform = ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.2))

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/f_dist.pdf")

        return fig

    def _plot_mt3_dist(self):
        fig, axs = plt.subplots(1, 2, sharey = 'row')
        x_min = self.inj_s.min() - 75
        x_max = self.inj_s.max() + 75
        y_min = self.p_i.min()
        y_max = self.p_i.max()

        axs[0].hist2d(self.inj, self.p_i, cmap = 'plasma', bins = (100, 100), range = [[x_min, x_max], [y_min, y_max]])
        axs[0].set_title("M/T (initial)")
        axs[0].set_xlabel("Injection time (ns)")
        axs[0].set_ylabel("Momentum (GeV)")

        # hist2d_final = axs[1].hist2d(self.inj, self.p, cmap = 'plasma', bins = (100, 100), range = [[x_min, x_max], [y_min, y_max]])
        # axs[1].set_title("M/T (final)")
        # axs[1].set_xlabel("Injection time (ns)")

        # Range specification needed since the axis limits are based on the rightmost plot (for some reason).
        # This weird bug? feature? only seems to happen with hist2d (not with the momentum distributions from above)
        hist2d_saved = axs[1].hist2d(self.inj_s, self.p_s, cmap = 'plasma', bins = (100, 100), range = [[x_min, x_max], [y_min, y_max]])
        axs[1].set_title("M/T (surviving)")
        axs[1].set_xlabel("Injection time (ns)")
        
        fig.colorbar(hist2d_saved[3], ax = axs)

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/mt3_dist.pdf")

        return fig

    def _plot_mt1_dist(self):
        offset_s_avg, offset_s_std = Plotter._calculate_stats(self.offset_s)
        pulse_length = self.inj_s.max() - self.inj_s.min()

        fig, ax = plt.subplots()
        ax.set_title(f"Offset / injection time distribution")
        ax.set_ylabel("Radial offset (mm)")
        ax.set_xlabel("Injection time (ns)")
        ax.set_yticks(ticks = [-45, -30, -15, 0, 15, 30, 45])
        ax.set_xticks(ticks = [-200, -175, -150, -125, -100, -75, -50, -25, 0, 25, 50, 75])             # hard coded for run2 injection profiles
        hist, xedges, yedges, image = ax.hist2d(self.inj_s, self.offset_s, range = [[-self.t_align - 100, -self.t_align + 100], [-45, 45]], cmap = 'plasma', bins = (200, 150))
        
        xcenters = [(xedges[i] + xedges[i+1])/2 for i in range(len(xedges)-1)]                          # plotting the mean radial offset as a function of inj time
        ycenters = [(yedges[i] + yedges[i+1])/2 for i in range(len(yedges)-1)]
        
        weights = [hist[i][:] for i in range(len(xcenters))]                                            # the weights are from the histogram bins
        clean_weights = [weight if np.sum(weight) != 0 else np.ones(shape = len(weight)) for weight in weights]
        yavg = [np.average(ycenters, weights = clean_weight) for clean_weight in clean_weights]         # calc the mean radial offset weighted by the (cleaned) histogram bins
        trimmed_yavg = np.array(yavg) * [np.sum(weight) > 0.001*np.sum(hist) for weight in weights]     # hide the means from time slices outside the main MTP (uses original weights)
        masked_yavg, masked_xcenters = aux.mask_zeroes(trimmed_yavg, xcenters)
        
        ax.plot(masked_xcenters, masked_yavg, color = (0, 1, 1), label = 'Mean radial offset')                        # plot the data
        
        fig.colorbar(image)

        Plotter._compare(self._p_acc_dir, ax, self.ring.n, self.t_align)                                # momentum acceptance plot
        
        ax.legend()
        ax.text(0.75, 0.15, s = r"$\langle r \rangle$ [mm]" + f" = {offset_s_avg:.2f}\n$\sigma_r$ [mm] = {offset_s_std:.2f}\nstore rate \u2248 {self.store_percentage:.2f}%\npulse length [ns] \u2248 {pulse_length:.2f}", ha = 'left', va = 'top', ma = 'left', transform = ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.9))

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/mt1_dist.pdf")

        return fig
        
    def _plot_pulse(self):
        kfield = self.ring.b_k.b_k * 10_000
        kfield_t_list = self.ring.b_k.t_list * 1E9
        inj_heights, inj_edges = np.histogram(self.inj, bins = 1000)
        inj_centers = (inj_edges[:-1] + inj_edges[1:]) / 2
        
        fig, ax = plt.subplots()
        ax.set_title("Kick field and injection pulse")
        ax.set_ylabel("Kicker field (G)")
        ax.set_xlabel("Time (ns)")
        ax.plot(kfield_t_list, kfield, color = 'blue', label = "Kicker field")
        ax_pulse2 = ax.twinx()
        ax_pulse2.set_ylabel("Injection pulse (counts)")
        ax_pulse2.plot(inj_centers, inj_heights, color = 'red', label = "Injection pulse")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform = ax.transAxes)

        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/pulse.pdf")

        return fig

    def _plot_spatial(self):
        scaled_r_min = 5
        scaled_r_max = 15
        circle = np.linspace(0, 2 * np.pi, 360)
        scaled_r_0 = np.full(shape = len(circle), fill_value = ((scaled_r_min + scaled_r_max) / 2))
        scaled_r = Plotter._rescale(self.r, self.ring.r_min, self.ring.r_max, scaled_r_min, scaled_r_max)
        r_bin_edges = np.linspace(scaled_r_min, scaled_r_max, 30)

        fig, ax = plt.subplots(1, 1, subplot_kw = dict(projection = "polar"), constrained_layout = False)
        ax.plot(circle, scaled_r_0, linestyle = 'dotted', color = 'yellow')
        ax.set_title("Muon spatial distribution at sim end")
        ax.spines['polar'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        spatial = ax.hist2d(self.phi, scaled_r, bins = [circle, r_bin_edges])        
        
        # ax.plot([aux.k1_i, aux.k1_i], [scaled_r_min, scaled_r_max])   # kicker region
        # ax.plot([aux.k3_f, aux.k3_f], [scaled_r_min, scaled_r_max])

        fig.colorbar(spatial[3], ax = ax)
        ax.set_rmin(0)

        self._labels(fig)

        self._merge(fig, f"{self.output_path}/spatial.pdf")

        return fig
    
    def _plot_inj(self):
        windowed_bins = 40
        inj_s, edges = np.histogram(self.inj_s, bins = windowed_bins)
        inj_windowed, edges_windowed = np.histogram(self.inj, bins = windowed_bins, range = (edges[0], edges[-1]))
        centers_windowed = (edges[1:] + edges[:-1]) / 2
        store_rate = inj_s / inj_windowed * 100
        
        full_bins = int(windowed_bins * (self.inj.max() - self.inj.min()) / (self.inj_s.max() - self.inj_s.min()))
        inj_full, edges_full = np.histogram(self.inj, bins = full_bins)
        centers_full = (edges_full[1:] + edges_full[:-1]) / 2

        fig, ax = plt.subplots()
        ax.set_title("Injection beam")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Injection time [ns]")
        ax.plot(centers_full, inj_full, color = 'blue', label = "Incoming")
        ax.plot(centers_windowed, inj_s, color = 'red', label = "Outgoing")

        ax2 = ax.twinx()
        ax2.plot(centers_windowed, store_rate, color = 'black', label = "Incoming / outgoing", alpha = 0.7)
        ax2.set_ylabel("Store rate [%]")

        ax1_ylims = ax.axes.get_ylim()
        ax1_yratio = ax1_ylims[0] / ax1_ylims[1]
        ax2_ylims = ax2.axes.get_ylim()
        ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
        if ax1_yratio < ax2_yratio: 
            ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
        else:
            ax.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform = ax.transAxes)

        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/inj.pdf")

        return fig

    def _plot_x_alpha(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        range = [[-10, 10], [-10, 10]]

        ax1.hist2d((self.r_init - aux.r_inj)*1000, self.alpha, bins = (40, 40), range = range, cmap = 'plasma')
        ax1.set_title("x-injection angle (all)")
        ax1.set_xlabel("Injection location (distance from injector center) [mm]")
        ax1.set_ylabel("Injection angle [mrad]")

        ax2.hist2d((self.r_init_s - aux.r_inj)*1000, self.alpha_s, bins = (40, 40), range = range, cmap = 'plasma')
        ax2.set_title("x-injection angle (stored)")
        ax2.set_xlabel("Injection location (distance from injector center) [mm]")
        ax2.set_ylabel("Injection angle [mrad]")

        self._labels(fig)
        
        self._merge(fig, f"{self.output_path}/x_alpha.pdf")

        return fig
    
    def _plot_stats(self):
        n = self.ring.n
        subruns, avg_x, sigma_x, C_E, store_rate = np.loadtxt(f"{self.data_path}/stats.txt", skiprows = 1, usecols = (0, 1, 2, 3, 4), unpack = True)
        unc_mu = avg_x / np.sqrt(self.muons)
        unc_sigma = sigma_x / np.sqrt(2 * self.muons - 2)
        unc_store = np.sqrt(store_rate * (1 - store_rate) / self.muons)
        unc_C_E = 4 * n * (1 - n) * aux.beta_magic**2 / aux.r_magic**2 * np.sqrt(unc_mu**2 * avg_x**2 + unc_sigma**2 * sigma_x*2 + 2 * avg_x * sigma_x)
        plex_var = self.labels[0].split("=")[0]
        xvals = [label.split("=")[1] for label in self.labels]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # avg_line = ax1.errorbar(xvals, avg_x, yerr = unc_mu, label = r"$\langle x \rangle$", color = 'red')
        avg_line = ax1.plot(xvals, avg_x, label = r"$\langle x \rangle$", color = 'red')
        # sigma_line = ax1.errorbar(xvals, sigma_x, yerr = unc_sigma, label = r"$\sigma_x$", color = 'blue')
        sigma_line = ax1.plot(xvals, sigma_x, label = r"$\sigma_x$", color = 'blue')
        ax1.set_xlabel(plex_var)
        ax1.set_ylabel("[mm]")
        ax3 = ax1.twinx()
        C_E_line = ax3.plot(xvals, C_E, label = "C_E", color = 'black')
        ax3.set_ylabel("[ppb]")
        print(avg_line)
        lines = avg_line + sigma_line + C_E_line
        print(lines)
        # lines = [avg_line.lines[0], sigma_line.lines[0], C_E_line]
        ax1.legend(lines, [l.get_label() for l in lines])

        # ax2.errorbar(xvals, store_rate, yerr = unc_store)
        ax2.set_title("Store rate")
        ax2.set_ylabel("[%]")
        ax2.set_xlabel(plex_var)

        pdf = PdfPages(f"{self.output_path}/stats.pdf")
        pdf.savefig(fig)
        pdf.close()

        return fig

    # endregion

    def _merge(self, fig, plot_path):
        """Merge a Matplotlib figure into a PDF."""
        # Save the individual plot into a PDF, and close it.
        plot = PdfPages(plot_path)
        plot.savefig(fig)
        plot.close()

        # Extract the overall plot PDF and the individual plot PDF.
        all_plots_pdf = pypdf.PdfFileReader(open(self.pdf_path, 'rb'))
        plot = pypdf.PdfFileReader(open(plot_path, 'rb'))

        # Initialize the PdfFileMerger object, and merge both PDFs into it.
        merger = pypdf.PdfFileMerger(strict=True)

        merger.append(all_plots_pdf)
        merger.append(plot)
        merger.write(self.pdf_path)
        merger.close()
        os.remove(plot_path)

    def _prep(self):
        single_muon_plots = ["trajectory", "offset", "phi", "vr", "vphi", "p", "kicker"]
        multi_muon_plots = ["p_dist", "f", "mt3", "mt1", "pulse", "spatial", "inj", "mt_ratio", "x_alpha", "stats"]
        if self.muons == 1:
            if set(self.multiplex_plots).issubset(set(single_muon_plots)):
                self.plot_funcs = {}
                for plot in self.multiplex_plots:
                    self.plot_funcs[plot] = self._ID_plotting_func(plot)
            else:
                warnings.warn(message = f"Given your single muon simulations, the following multiplex plots will be ignored: {list(set(self.multiplex_plots).difference(single_muon_plots))}", category = UserWarning, stacklevel = 3)
                self.multiplex_plots = list(set(self.multiplex_plots) & set(single_muon_plots))
                self.plot_funcs = {}
                for plot in self.multiplex_plots:
                    self.plot_funcs[plot] = self._ID_plotting_func(plot)
        else:
            if set(self.multiplex_plots).issubset(set(multi_muon_plots)):
                self.plot_funcs = {}
                for plot in self.multiplex_plots:
                    self.plot_funcs[plot] = self._ID_plotting_func(plot)
            else:
                print(f"Given your multi muon simulations, the following multiplex plots will be ignored: {list(set(self.multiplex_plots).difference(multi_muon_plots))}")
                self.multiplex_plots = list(set(self.multiplex_plots) & set(multi_muon_plots))
                self.plot_funcs = {}
                for plot in self.multiplex_plots:
                    self.plot_funcs[plot] = self._ID_plotting_func(plot)
    
    def _extract(self, data_path):
        """Extract the relevant results from the .npz file, setting the number of muons as well."""
        try:
            results = np.load(data_path + "/states.npz")
            self.number = 1 # single muons
            r = results['r'] # meters
            phi = results['phi'] # radians
            vr = results['vr']  # m/s
            vphi = results['vphi'] # rad/s

            self.p = aux.state_to_mom_cyl(r = r, vr = vr, vphi = vphi)
            self.r = r # m
            self.phi = phi # rad
            self.vr = vr * 1E-6 # mm/ns
            self.vphi = vphi * 1E-9 / np.pi * 180 # deg/ns
            self.t = results['t'] * 10**9 # ns

            self.r_offset = (r - ((self.ring.r_min + self.ring.r_max) / 2)) * 1000 # mm
        except FileNotFoundError:
            final_states = np.load(data_path + "/final_states.npz")
            init_states = np.load(data_path + "/init_states.npz")
            self.number = 2 # multiple muons

            lost = final_states["lost"]
            self.r = final_states["r"]
            self.r_init = init_states['r']
            self.phi = final_states["phi"] % (2 * np.pi)
            self.store_percentage = (1 - len(np.where(lost == True)[0]) / len(lost)) * 100

            self.p_i = aux.state_to_mom_cyl(r = init_states["r"], vr = init_states["vr"], vphi = init_states["vphi"])
            self.p = aux.state_to_mom_cyl(r = final_states["r"], vr = final_states["vr"], vphi = final_states['vphi'])
            self.f = aux.p_to_f(momentum = self.p, n = self.ring.n)
            self.inj = init_states["t"] * 10**9
            self.alpha = np.arctan(init_states['vr'] / init_states['vphi'] / init_states['r']) * 1000
            self.p_s, self.f_s, self.inj_s, self.alpha_s, self.r_init_s= Plotter._mask_lost(lost, self.p, self.f, self.inj, self.alpha, self.r_init)
            self.offset_s = aux.p_to_rco(p = self.p_s, n = self.ring.n)

    def _kicker_regions(self, ax, phi, t):
        """Highlight the kicker regions in a plot; the three individual kickers are combined."""
        entrance_times, exit_times = Plotter._find_in_out_times(phi, aux.k1_i, aux.k3_f, t, self.ring.b_k.kick_max)
        legend = False
        for i in range(len(exit_times)):
            ax.axvspan(entrance_times[i], exit_times[i], color = 'red', alpha = 0.3, label = 'Kicker region' if legend == False else "")

            legend = True

    def _quad_regions(self, ax, phi, t):
        """Highlight the quadruple regions in a plot; long and short segemnts are combined."""
        if self.ring.quad_num == 4:
            legend = False
            for quad in aux.quad_lims:
                entrance_times, exit_times = Plotter._find_in_out_times(phi, aux.quad_lims[quad][0], aux.quad_lims[quad][1], t, self.ring.b_k.kick_max)

                for i in range(len(exit_times)):
                    ax.axvspan(entrance_times[i], exit_times[i], color = 'yellow', alpha = 0.3, label = f'ESQ (n = {self.ring.n})' if legend == False else "")
                    legend = True
    
    def _labels(self, fig):
        if self.labels:
            fig.suptitle(self.current_label)

    def _find_plottable_subruns(self, subruns):
        self.plottable_subruns = []
        if self.muons == 1:
            self.plottable_subruns = [1, 2]
            # for i in range(subruns):
            #     subrun = i+1
            #     lost = np.load(f"{self.data_path}/subrun_{subrun}_{subruns}/final_states.npz")['lost']
            #     if sum(lost) != len(lost):
            #         self.plottable_subruns.append(subrun)
            #         print(f"Plottable subruns not given, searching... Subrun ID'd: {subrun}", end = "\r")
        else:
            for i in range(subruns):
                subrun = i+1
                lost = np.load(f"{self.data_path}/subrun_{subrun}_{subruns}/final_states.npz")['lost']
                if sum(lost) != len(lost):
                    self.plottable_subruns.append(subrun)
                    print(f"Plottable subruns not given, searching... Subrun ID'd: {subrun}", end = "\r")

    def _ID_plotting_func(self, plot):
        if plot == "trajectory":
            return self._plot_trajectory
        elif plot == "offset":
            return self._plot_offset
        elif plot == "phi":
            return self._plot_phi
        elif plot == "vr":
            return self._plot_vr
        elif plot == "vphi":
            return self._plot_vphi
        elif plot == "p":
            return self._plot_p
        elif plot == "kicker":
            return self._plot_kicker

        elif plot == "p_dist":
            return self._plot_p_dist
        elif plot == "f":
            return self._plot_f_dist
        elif plot == "mt3":
            return self._plot_mt3_dist
        elif plot == "mt1":
            return self._plot_mt1_dist
        elif plot == "pulse":
            return self._plot_pulse
        elif plot == "spatial":
            return self._plot_spatial
        elif plot == "inj":
            return self._plot_inj
        elif plot == "x_alpha":
            return self._plot_x_alpha
        elif plot == "stats":
            return self._plot_stats

    @staticmethod
    def _find_in_out_times(phi, init_edge, final_edge, t, kick_max):
        phi = phi % ( 2 * np.pi)
        bools = np.logical_and(phi > init_edge, phi < final_edge)

        enters = []
        exits = []
        start = 0

        while start < len(bools):
            ind_enter = np.argmax(bools[start:] == True)
            if ind_enter != 0:
                enters.append(start + ind_enter)
            else:
                break
            ind_exit = np.argmax(bools[start + ind_enter:] == False)
            if ind_exit != 0:
                exits.append(start + ind_enter + ind_exit)
            else:
                break
            start = start + ind_enter + ind_exit
        if len(enters) != len(exits):
            exits.append(-1)
        
        in_times = t[enters]
        out_times = t[exits]
        if kick_max > len(in_times):
            return in_times, out_times
        else:
            return in_times[:kick_max], out_times[:kick_max]

    @staticmethod
    def _rescale(r, r_min, r_max, scaled_min, scaled_max):
        """Rescale radial data to more visually-apparent lengths."""
        return scaled_min + (r - r_min) / (r_max - r_min) * (scaled_max - scaled_min)
    
    @staticmethod
    def _mask_lost(lost, *data):
        """Mask the lost muons from the data."""
        indices = np.where(lost == True)[0]
        if len(data) == 1:
            masked_array = np.delete(data[0], obj = indices, axis = 0)
            return masked_array
        else:
            masked_data = []
            for array in data:
                masked_array = np.delete(array, obj = indices, axis = 0)
                masked_data.append(masked_array)
            return tuple(masked_data)
    
    @staticmethod
    def _calculate_stats(*data):
        """Calculate the mean and standard deviation of the given data."""
        if len(data) == 1:
            return data[0].mean(), data[0].std(ddof = 1)
        else:
            stats = ()
            for array in data:
                stats += ((array.mean(), array.std(ddof = 1)),)
            return stats
        
    @staticmethod
    def _compare(dir, ax, n, t_align):
        """Compare the M/T histogram with the results of a momentum acceptance simulation."""
        if dir is None or dir == "":
            pass
        else:
            try:
                path = str(pathlib.Path(__file__).parent.absolute()) + f"/momentum_acceptance_data/{dir}/captured_momenta_out.dat"
                if dir in ["n0k1", "n0k2", "n0k3", "n0k4", "n1k1", "n1k2", "n1k3", "n1k4"]:
                    t, dpmax, dpmin = np.loadtxt(fname = path, skiprows = 2, usecols = (0, 6, 7), unpack = True)
                    dpmax, dpmin, t = aux.mask_zeroes(dpmax, dpmin, t)
                    t = t - t_align                                    # convert to injection time
                    pmax = dpmax * (aux.r_magic / (1 - n) * 1000)
                    pmin = dpmin * (aux.r_magic / (1 - n) * 1000)
                    ax.plot(t, pmin, color = 'red', linewidth = 3, label = 'window')
                    ax.plot(t, pmax, color = 'red', linewidth = 3)
                    ax.legend(loc = "upper right")

                else:
                    t, centroid, halfwidth = np.loadtxt(fname = path, skiprows = 1, usecols = (0, 4, 6), unpack = True)
                    centroid, halfwidth, t = aux.mask_zeroes(centroid, halfwidth, t) # mask the times and halfwidths according to the centroid
                    t = t - 37.5 # convert to injection time
                    centroid *= (aux.r_magic / (1 - n) * 1000)
                    halfwidth *= (aux.r_magic / (1 - n) * 1000)
                    ax.plot(t, centroid - halfwidth, color = 'red', linewidth = 3, label = 'window')
                    ax.plot(t, centroid + halfwidth, color = 'red', linewidth = 3)
                    ax.legend(loc = "upper right")
                
            except (OSError, FileNotFoundError):
                warnings.warn(message = "Your momentum acceptance data could not be parsed, and so will be ignored.", category = UserWarning, stacklevel = 2)

class AlignPlotter:
    def __init__(self, dir):
        self.full_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + dir
        self.coarse_t0s, self.coarse_rates = np.loadtxt(f"{self.full_path}/coarse.txt", skiprows = 1, usecols = (1, 2), unpack = True)
        self.fine_t0s, self.fine_rates = np.loadtxt(f"{self.full_path}/fine.txt", skiprows = 1, usecols = (1, 2), unpack = True)
        self._fit()
        with open(f"{self.full_path}/aux.txt", "r") as file:
            self.inj, kicker = file.readlines()
        self.inj = self.inj.strip()
        self.kicker, self.b_norm, self.t_norm = kicker.split(', ')
        (self.kicker_t, self.kicker_strength), (self.injection_t, self.injection_heights) = self._load_pulses()
    
    def settings(self):
        # if self.plot_option == "presentation":
        plt.rcParams["figure.figsize"] = (8,6)
        plt.rcParams["figure.titlesize"] = 9
        plt.rcParams["axes.titlesize"] = 17
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams['axes.formatter.useoffset'] = False
        # elif self.plot_option == "maximize_graph":
        #     plt.rcParams["figure.figsize"] = (21,10)
        #     plt.rcParams["figure.titlesize"] = 9
        #     plt.rcParams["axes.titlesize"] = 15
        #     plt.rcParams["axes.labelsize"] = 12
        #     plt.rcParams["font.size"] = 14
        
        # plt.rcParams["xtick.labelsize"] = 10
        # plt.rcParams["ytick.labelsize"] = 10
        
        plt.rcParams["font.family"] = "sans-serif"
        
        plt.rcParams["image.cmap"] = 'plasma'
        plt.rcParams['figure.constrained_layout.use'] = True
        # plt.rcParams["figure.autolayout"] = True
        plt.rcParams["text.usetex"] = False
        plt.rcParams["axes.formatter.use_mathtext"] = True

        # Make axis tick marks face inward.
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        # Draw axis tick marks all around the edges.
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True

        # Draw minor axis tick marks in between major labels.
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True

        # Make all tick marks longer.
        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["xtick.minor.size"] = 4
        plt.rcParams["ytick.major.size"] = 8
        plt.rcParams["ytick.minor.size"] = 4
    
    def plot(self):
        self.settings()
        self._create_figures()
        self._save()
        
    def _create_figures(self):
        self.fig_coarse, ax_coarse = plt.subplots(1, 1)
        ax_coarse.plot(self.coarse_t0s, self.coarse_rates)
        ax_coarse.set_title("Coarse scan")
        ax_coarse.set_xlabel("t' [ns]")
        ax_coarse.set_ylabel("store rate [%]")
        
        self.fig_fine, ax_fine = plt.subplots(1, 1)
        ax_fine.plot(self.fine_t0s, self.fine_rates)
        ax_fine.set_title("Fine scan")
        ax_fine.set_xlabel("t' [ns]")
        ax_fine.set_ylabel("store rate [%]")
        
        def quad(xs, a, b, c):
            return np.array([a*x*x + b*x + c for x in xs])

        xs = np.linspace(start = self.fine_t0s[self.idx-1], stop = self.fine_t0s[self.idx+1], num = 50)
        ys = quad(xs, self.coeffs[0], self.coeffs[1], self.coeffs[2])
        ax_fine.plot(xs, ys, color = 'gray', linestyle = '--', alpha = 0.5)
        ax_fine.axvline(x = self.final_t0, color = "red", linewidth = 0.5)
        ax_fine.axhline(y = self.final_store, color = "red", linewidth = 0.5)

        # transh = transforms.blended_transform_factory(ax_fine.get_yticklabels()[0].get_transform(), ax_fine.transData)
        # transv = transforms.blended_transform_factory(ax_fine.get_xticklabels()[0].get_transform(), ax_fine.transData)
        ax_fine.text(0.8, 0.97, s = f"t': {self.final_t0:5.2f} ns\nstore rate: {self.final_store:5.2f}%", ha = 'left', va = 'top', ma = 'left', transform = ax_fine.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'gray', alpha = 0.2))

        ax_fine.tick_params(axis='x', labelrotation = 45, labelsize = 8)

        self.fig_pulse, ax_pulse = plt.subplots(1,1)
        k_line = ax_pulse.plot(self.kicker_t, self.kicker_strength, color = 'blue', label = "Kicker strength")
        ax_pulse2 = ax_pulse.twinx()
        inj_line = ax_pulse2.plot(self.injection_t, self.injection_heights, color = 'red', label = "Injection time")
        kick_line = ax_pulse2.plot(self.injection_t + 37.5, self.injection_heights, color = 'orange', label = "Kick time")
        lines = k_line + inj_line + kick_line
        ax_pulse.legend(lines, [l.get_label() for l in lines])
        ax_pulse.set_title("Injection and kicker pulses")
        ax_pulse.set_xlabel("Time [ns]")
        ax_pulse.set_ylabel("Kicker strength [G]")
        ax_pulse2.set_ylabel("Injection frequency [arb]")


    def _save(self):
        pdf_coarse = PdfPages(self.full_path + "/coarse.pdf")
        pdf_fine = PdfPages(self.full_path + "/fine.pdf")
        pdf_pulse = PdfPages(self.full_path + "/pulse.pdf")

        pdf_coarse.savefig(self.fig_coarse)
        pdf_fine.savefig(self.fig_fine)
        pdf_pulse.savefig(self.fig_pulse)
        pdf_coarse.close()
        pdf_fine.close()
        pdf_pulse.close()

        # Extract both plots.
        pdf_coarse = pypdf.PdfFileReader(open(self.full_path + "/coarse.pdf", 'rb'))
        pdf_fine = pypdf.PdfFileReader(open(self.full_path + "/fine.pdf", 'rb'))
        pdf_pulse = pypdf.PdfFileReader(open(self.full_path + "/pulse.pdf", 'rb'))

        # Initialize the PdfFileMerger object, and merge both PDFs into it.
        merger = pypdf.PdfFileMerger(strict=True)

        merger.append(pdf_coarse)
        merger.append(pdf_fine)
        merger.append(pdf_pulse)
        merger.write(self.full_path + "/plots.pdf")
        merger.close()
        os.remove(self.full_path + "/coarse.pdf")
        os.remove(self.full_path + "/fine.pdf")
        os.remove(self.full_path + "/pulse.pdf")

    def _fit(self):
        self.idx = np.argmax(self.fine_rates)
        self.coeffs = np.polyfit(self.fine_t0s[self.idx-1:self.idx+2], self.fine_rates[self.idx-1:self.idx+2], deg = 2)
        a, b, c = self.coeffs
        self.final_t0, self.final_store = -b / (2 * a), c - (b**2 / 4 / a)
    
    def _load_pulses(self):
        source = str(pathlib.Path(__file__).parent.absolute()) + "/injection_profiles/" + self.inj + ".dat"
        bin_centers, bin_heights = np.loadtxt(fname = source, skiprows = 5, usecols = (0, 1), unpack = True)
        bin_heights_masked = bin_heights[bin_heights >= 0]
        bin_centers = bin_centers[bin_heights >= 0] - self.final_t0 # cut the bin centers array as well
        injection = (bin_centers, bin_heights_masked)

        dir = str(pathlib.Path(__file__).parent.absolute()) + "/kicker_pulses/" + self.kicker + ".dat"
        t, b = np.loadtxt(fname = dir, skiprows = 1, usecols = (0, 1), unpack = True)
        
        # Mask unnecessary parts of the field.
        b, t = aux.mask_zeroes(b, t)

        # Normalize strength and timing if necessary.
        if self.b_norm != "":
            b *= (float(self.b_norm) / b.max())
        t = (t - float(self.t_norm))
        t_mask = t[t < bin_centers[-1]]
        b = b[t < bin_centers[-1]]
        kicker = (t_mask, b)

        return kicker, injection