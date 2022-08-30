import numpy as np
import gm2mt.auxiliary as aux
import gm2mt.Distributions as dist
import pandas as pd

import pathlib
import warnings
import copy

class StateGenerator:
    def __init__(
        self,
        mode = "mc",
        seed = None,
        muons = 1,
        momentum = dist._NoneDist(),        # Momentum (GeV)
        f = dist._NoneDist(),               # Frequency (kHz)
        offset = dist._NoneDist(),          # Offset (proxy for momentum; mm)
        initial_offset = dist._NoneDist(),  # Initial radial offset (mm)
        alpha = dist._NoneDist(),           # Injection angle (mrad)
        x_alpha = dist._NoneDist(),         # Initial offset [mm] and Injection angle [mrad] (correlated)
        phi_0 = 0,                          # Initial anglular position (rad)
        t = 0                               # Injection time (ns)
    ):
        if mode == "bmad" and isinstance(seed, str):
            self.mode = mode
            self.seed = seed

            if isinstance(muons, (int, float)):
                self.muons = int(muons)
            else:
                warnings.warn(f"Your muon number '{muons}' is invalid but will be overwritten.", SyntaxWarning, stacklevel = 2)

            if isinstance(t, (float, int)):
                self.t = dist.Single(value = t)
            else:
                self.t = t
            
            self.MPIDs = {'t': self.t._MPID}
            mp_count = 0
            for MPID in self.MPIDs.values():
                mp_count += MPID['MPs']

            if mp_count == 0:
                self.plex = "simplex"
            elif mp_count == 1:
                self.plex = "multiplex"
            else:
                raise RuntimeError("The multiplex system cannot handle more than one changing variable; please change!")
        elif mode == "mc" and (isinstance(seed, int) or seed is None):
            self.mode = mode
            self.seed = seed

            if isinstance(muons, (float, int)):
                self.muons = int(muons)
            else:
                raise TypeError("The number of muons specified is the wrong type!")
            
            if isinstance(x_alpha, (dist.Custom2D, dist._NoneDist)):
                self.x_alpha = x_alpha
            elif isinstance(x_alpha, list):
                for x_alpha_dist in x_alpha:
                    if not isinstance(x_alpha_dist, (dist.Custom2D, dist._NoneDist)):
                        raise ValueError(f"Your x_alpha parameter (type: {type(x_alpha)}) is not type dist.Custom2D")
                self.x_alpha = x_alpha
            else:
                raise ValueError(f"Your x_alpha parameter (type: {type(x_alpha)}) is not type dist.Custom2D")
            
            if isinstance(initial_offset, (float, int)):
                self.initial_offset = dist.Single(value = initial_offset)
            else:
                self.initial_offset = initial_offset

            if isinstance(alpha, (float, int)):
                self.alpha = dist.Single(value = alpha)
            else:
                self.alpha = alpha
            
            if isinstance(self.x_alpha, dist.Custom2D):
                if not isinstance(self.initial_offset, dist._NoneDist) or not isinstance(self.alpha, dist._NoneDist):
                    raise ValueError("You have overspecified the initial offsets and injection angles")
                else:
                    self.xa_specifier = "correlated"
            else:
                if isinstance(self.initial_offset, dist._NoneDist) or isinstance(self.alpha, dist._NoneDist):
                    raise ValueError("You have underspecified the initial offsets and injection angles")
                else:
                    self.xa_specifier = "independent"
            
            if isinstance(phi_0, (float, int)):
                self.phi_0 = dist.Single(value = phi_0)
            else:
                self.phi_0 = phi_0
            
            if isinstance(t, (float, int)):
                self.t = dist.Single(value = t)
                self.zero = t
            elif isinstance(t, dist.Single):
                self.t = t
                self.zero = t.value
            else:
                self.t = t
                self.zero = t.zero[0]       # this (might) work if t is type dist.Custom
            
            none_count = sum([isinstance(i, dist._NoneDist) for i in [momentum, offset, f]])
            if none_count == 3:
                raise SyntaxError("You have underspecified the muon speed!")
            elif none_count == 1:
                raise SyntaxError("You have overspecified the muon speed!  Please pick one and leave the others as None.")

            self.momentum = dist._NoneDist()
            self.offset = dist._NoneDist()
            self.f = dist._NoneDist()

            if not isinstance(momentum, dist._NoneDist):
                if isinstance(momentum, (float, int)):
                    self.momentum = dist.Single(value = momentum)
                else:
                    self.momentum = momentum
                self.v_specifier = "momentum"
            elif not isinstance(offset, dist._NoneDist):
                if isinstance(offset, (float, int)):
                    self.offset = dist.Single(value = offset)
                else:
                    self.offset = offset
                self.v_specifier = "offset"
            elif not isinstance(f, dist._NoneDist):
                if isinstance(f, (float, int)):
                    self.f = dist.Single(value = f)
                else:
                    self.f = f
                self.v_specifier = "f"

            self.MPIDs = {'p': self.momentum._MPID, 'f': self.f._MPID, 'r_co': self.offset._MPID, 'x': self.initial_offset._MPID, 'alpha': self.alpha._MPID, 'x_alpha': self.x_alpha._MPID, 'phi_0': self.phi_0._MPID, 't': self.t._MPID}
            mp_count = 0
            for MPID in self.MPIDs.values():
                mp_count += MPID['MPs']

            if mp_count == 0:
                self.plex = "simplex"
            elif mp_count == 1:
                self.plex = "multiplex"
            else:
                raise RuntimeError("The multiplex system cannot handle more than one changing variable; please change!")
        else:
            raise ValueError(f"Your mode '{mode}' and seed '{seed}' are not valid.")
        
    def generate(self, integration_method, n, verbose = True):
        self.integration_method = integration_method
        if self.mode == "bmad" and self.plex == "simplex":
            return self._generate_bmad_simplex(verbose)
        if self.mode == "bmad" and self.plex == "multiplex":
            return self._generate_bmad_multiplex(verbose)
        elif self.mode == "mc" and self.plex == "simplex":
            return self._generate_mc_simplex(n, verbose)
        elif self.mode == "mc" and self.plex == "multiplex":
            return self._generate_mc_multiplex(n, verbose)

    def _generate_mc_simplex(self, n, verbose = True):
        self.loops = 1
        self.labels = []
        # Use MC generators to generate the initial conditions, and convert to the appropriate SI units.
        # The only exception is the momentum, which is left as is in GeV.

        bit_generator = np.random.MT19937(seed = self.seed)
        rng = np.random.default_rng(seed = bit_generator)
        # initial_offset = self.initial_offset.select(number = self.muons, rng = rng) / 10**3 # from mm to m
        # r = initial_offset + aux.r_magic # no conversion since initial_offset has already been converted
        # alpha = self.alpha.select(number = self.muons, rng = rng) / 1000 # from mrad to rad

        if self.v_specifier == "momentum":
            p = self.momentum.select(number = self.muons, rng = rng) / 100 * aux.p_magic # convert from % to GeV
            v = aux.p_to_v(momentum = p)
        elif self.v_specifier == "offset":
            offset = self.offset.select(number = self.muons, rng = rng)
            v = aux.rco_to_v(offset, n)
        elif self.v_specifier == "f":
            f = self.f.select(number = self.muons, rng = rng)
            v = aux.f_to_v(f, n)

        if self.xa_specifier == "correlated":
            x, alpha = self.x_alpha.select(number = self.muons, rng = rng)
        elif self.xa_specifier == "independent":
            x = self.initial_offset.select(number = self.muons, rng = rng)
            alpha = self.alpha.select(number = self.muons, rng = rng)
        
        r = x/1E3 + aux.r_magic                 # from offset [mm] to radial pos [m]
        alpha = alpha/1E3                       # from [mrad] to [rad]

        phi_0 = self.phi_0.select(number = self.muons, rng = rng)       # no conversion
        t = self.t.select(number = self.muons, rng = rng) / 10**9       # from ns to s
        
        state = np.empty(shape = (self.muons, 5), dtype = np.float64)
        if self.integration_method == ("rk4" or "rkn"):
            state[:,0] = r
            state[:,1] = phi_0
            state[:,2] = v * np.sin(alpha)
            state[:,3] = v * np.cos(alpha) / r
            state[:,4] = t
        elif self.integration_method == "optical":
            raise ValueError("this isnt done yet")

        if verbose:
            print("Initial state(s) generated.")
        return state

    def _generate_mc_multiplex(self, n, verbose = True):
        # Use MC generators to generate the initial conditions, and convert to the appropriate SI units.
        # The only exception is the momentum, which is left as is in GeV.

        # Identify the changing parameter for this multiplex simulation:
        self.loops, self.labels = self.search()

        bit_generator = np.random.MT19937(seed = self.seed)
        rng = np.random.default_rng(seed = bit_generator)

        # initial_offset = self.initial_offset.mselect(number = self.muons, rng = rng, loops = self.loops) / 10**3 # from mm to m
        # r = initial_offset + aux.r_magic # no conversion since initial_offset has already been converted
        # alpha = self.alpha.mselect(number = self.muons, rng = rng, loops = self.loops) / 1000 # from mrad to rad

        if self.v_specifier == "momentum":
            p = self.momentum.mselect(number = self.muons, rng = rng, loops = self.loops) / 100 * aux.p_magic # convert from % to GeV
            v = aux.p_to_v(momentum = p)
        elif self.v_specifier == "offset":
            offset = self.offset.mselect(number = self.muons, rng = rng, loops = self.loops)
            v = aux.rco_to_v(offset, n)
        elif self.v_specifier == "f":
            f = self.f.mselect(number = self.muons, rng = rng, loops = self.loops)
            v = aux.f_to_v(f, n)

        if self.xa_specifier == "correlated":
            x, alpha = self.x_alpha.mselect(number = self.muons, rng = rng, loops = self.loops)
        elif self.xa_specifier == "indepedent":
            x = self.initial_offset.mselect(number = self.muons, rng = rng, loops = self.loops)
            alpha = self.alpha.mselect(number = self.muons, rng = rng, loops = self.loops)

        r = x/1E3 + aux.r_magic             # from offset [mm] to radial pos [m]
        alpha = alpha/1E3                   # from [mrad] to [rad]

        phi_0 = self.phi_0.mselect(number = self.muons, rng = rng, loops = self.loops) # no conversion
        t = self.t.mselect(number = self.muons, rng = rng, loops = self.loops) / 10**9 # from ns to s
        
        state = np.empty(shape = (self.loops, self.muons, 5), dtype = np.float64)
        if self.integration_method == "rk4":
            state[:,:,0] = r
            state[:,:,1] = phi_0
            state[:,:,2] = v * np.sin(alpha)
            state[:,:,3] = v * np.cos(alpha) / r
            state[:,:,4] = t
        elif self.integration_method == "optical":
            raise ValueError("this isnt done yet")
        
        if verbose:
            print(f"Initial state(s) generated.  Total loops: {self.loops}")
        return state
    
    def _generate_bmad_simplex(self, verbose):
        self.loops = 1
        self.labels = []

        source = str(pathlib.Path(__file__).parent.absolute()) + "/bmad_gm2rs/" + self.seed + ".dat"
        x, alpha, dp = np.loadtxt(source, skiprows = 1, usecols = (3, 4, 8), unpack = True)
        r = x + aux.r_magic
        v = aux.p_to_v( (1 + dp) * aux.p_magic)

        if verbose and self.muons != len(x):
            self.muons = len(x)
            print(f"Overwriting muon number to {self.muons}.")
        
        bit_generator = np.random.MT19937(seed = None)
        rng = np.random.default_rng(seed = bit_generator)
        t = self.t.select(number = self.muons, rng = rng) / 10**9 # from ns to s

        state = np.empty(shape = (self.muons, 5), dtype = np.float64)
        if self.integration_method == "rk4":
            state[:,0] = r
            state[:,1] = 0
            state[:,2] = v * np.sin(alpha)
            state[:,3] = v * np.cos(alpha) / r
            state[:,4] = t
        print(f"Initial states extracted from {self.seed}.dat.")
        return state

    def _generate_bmad_multiplex(self, verbose):
        self.loops, self.labels = self.search()

        source = str(pathlib.Path(__file__).parent.absolute()) + "/bmad_gm2rs/" + self.seed + ".dat"
        x, alpha, dp = np.loadtxt(source, skiprows = 1, usecols = (3, 4, 8), unpack = True)
        self.muons = len(x)
        if verbose:
            print(f"Overwriting muon number to {self.muons}")
        r = x + aux.r_magic
        v = aux.p_to_v( (1 + dp) * aux.p_magic)
        bit_generator = np.random.MT19937(seed = None)
        rng = np.random.default_rng(seed = bit_generator)
        t = self.t.mselect(number = self.muons, rng = rng, loops = self.loops) / 10**9

        state = np.empty(shape = (self.loops, self.muons, 5), dtype = np.float64)
        if self.integration_method == "rk4":
            state[:,:,0] = r
            state[:,:,1] = 0
            state[:,:,2] = v * np.sin(alpha)
            state[:,:,3] = v * np.cos(alpha) / r
            state[:,:,4] = t
        print(f"Initial states extracted from {self.seed}.dat.  Total loops: {self.loops}")
        return state

    def search(self):
        variable = ""
        for var, MPID in self.MPIDs.items():
            if MPID['MPs'] == 1:
                variable = var
        if variable == "":
            raise RuntimeError("Could not find the multiplex variable.  Something went wrong :(")

        loops = self.MPIDs[variable]['loops']
        # print(self.MPIDs[variable]['labels'])
        labels = self._label_prefixing(self.MPIDs[variable]['labels'], variable)
        # print(self.MPIDs[variable]['labels'])
        return loops, labels
    
    def _label_prefixing(self, labels, variable):
        for subrun, label in enumerate(labels):
            # labels[subrun] = f"[{subrun+1}] {variable}: {label}"
            labels[subrun] = f"{variable}: {label}"
        return labels

    def _store_parameters(self, full_path):
        if self.mode == "mc":
            with open(full_path + "/params_readable.txt", "a") as file:
                file.write("INITIAL STATE PARAMETERS:\n")
                file.write(f"RNG seed ID: {self.seed}\n")
                file.write(f"Muons: {self.muons}\n")

                if self.v_specifier == "momentum":
                    file.write(f"Momentum (GeV): {self.momentum.params_txt()}\n")
                elif self.v_specifier == "offset":
                    file.write(f"Offset (momentum proxy) (mm): {self.offset.params_txt()}\n")
                elif self.v_specifier == "f":
                    file.write(f"Frequency (momentum proxy) (kHz): {self.f.params_txt()}\n")
                
                if self.xa_specifier == "correlated":
                    file.write(f"Initial offset (mm) and deflection angle (mrad): {self.x_alpha.params_txt()}\n")
                elif self.xa_specifier == "independent":
                    file.write(f"Initial radial offset (mm): {self.initial_offset.params_txt()}\n")
                    file.write(f"Deflection angle angle (mrad): {self.alpha.params_txt()}\n")
                
                file.write(f"Initial angle (radians): {self.phi_0.params_txt()}\n")
                file.write(f"Injection time (ns): {self.t.params_txt()}\n")
            
            params = {'mode': [self.mode, 0, 0], 'seed': [self.seed, 0, 0], 'muons': [self.muons, 0, 0]}
            for param in ['momentum', 'f', 'offset', 'initial_offset', 'alpha', 'x_alpha', 'phi_0', 't']:
                params[param] = getattr(self, param).params_xl()

        elif self.mode == "bmad":
            with open(full_path + "/params_readable.txt", "a") as file:
                file.write("INITIAL STATE PARAMETERS:\n")
                file.write(f"BMAD file ID: {self.seed}\n")
                file.write(f"Injection time (ns): {self.t.params_txt()}\n")
            params = {'mode': [self.mode, 0, 0], 'seed': [self.seed, 0, 0], 'muons': [self.muons, 0, 0], 't': getattr(self, 't').params_xl()}

        df = pd.DataFrame(params)
        try:
            with pd.ExcelWriter(full_path + '/parameters.xlsx', engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = 'state_generator', index = False)
        except FileNotFoundError:
            with pd.ExcelWriter(full_path + '/parameters.xlsx', mode = 'w') as writer:
                df.to_excel(writer, sheet_name = 'state_generator', index = False)

    @classmethod
    def load(cls, dir):

        def convert_to_list(value):
            if isinstance(value, str) and value[0] == "[":
                try:
                    return [float(i) for i in value[1:-1].split(", ")]
                except ValueError:
                    return [val.replace("'","") for val in value[1:-1].split(", ")]
            else:
                return value
        
        def translate(params_list):
            if params_list[0] == "none":
                return None
            elif params_list[0] == "gaussian":
                return dist.Gaussian(params_list[1], params_list[2])
            elif params_list[0] == "uniform":
                return dist.Uniform(params_list[1], params_list[2])
            elif params_list[0] == "single":
                return dist.Single(params_list[1])
            elif params_list[0] == "triangle":
                return dist.Triangle(params_list[1], params_list[2])
            elif params_list[0] == "custom":
                return dist.Custom(params_list[1], params_list[2])
            elif params_list[0] == "custom 2d":
                return dist.Custom2D(params_list[1])
            elif params_list[0] == "nonedist":
                return dist._NoneDist()

        df = pd.read_excel(str(pathlib.Path(__file__).parent.absolute()) + "/results/" + dir + "/parameters.xlsx", sheet_name = "state_generator", header = 0, engine = "openpyxl")
        df_conv = df.applymap(convert_to_list)

        if isinstance(df_conv['seed'][0], str):
            seed = df_conv['seed'][0]
        elif np.isnan(df_conv['seed'][0]):
            seed = None
        elif isinstance(df_conv['seed'][0], (np.float64, np.int64, int, float)):
            seed = int(df_conv['seed'][0])
        
        if df_conv['mode'][0] == "mc":
            return cls(mode = str(df_conv['mode'][0]),
            seed = seed,
            muons = df_conv['muons'][0].item(),
            momentum = translate(df_conv['momentum']),
            offset = translate(df_conv['offset']),
            f = translate(df_conv['f']),
            initial_offset = translate(df_conv['initial_offset']),
            alpha = translate(df_conv['alpha']),
            x_alpha = translate(df_conv['x_alpha']),
            phi_0 = translate(df_conv['phi_0']),
            t = translate(df_conv['t'])
        )
        elif df_conv['mode'][0] == "bmad":
            return cls(mode = str(df_conv['mode'][0]),
                seed = seed,
                muons = df_conv['muons'][0].item(),
                t = translate(df_conv['t'])
            )