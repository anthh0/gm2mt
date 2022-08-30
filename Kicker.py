import numpy as np
import warnings
import pathlib
import copy
import pandas as pd

import gm2mt.auxiliary as aux

class Kicker:
    def __init__(self, mode, B, b_norm = None, t_norm = 0, kick_max = 100, kicker_num = 3):
        self.mode = mode
        self.B = B
        self.b_norm = b_norm
        
        self.t_norm = t_norm
        self.kick_max = kick_max
        self.kicker_num = kicker_num

        self.kicker_params = copy.deepcopy(vars(self))

        self.mp_mode = "simplex"
        # print(type(self.b_norm))
        for value in self.kicker_params.values():
            # print(value)
            # print(type(value))
            if isinstance(value, (list, np.ndarray)) and len(value) != 1:
                self.mp_mode = "multiplex"

        if self.mp_mode == "simplex":
            self._verify()
        
    def _verify(self):
        if self.mode not in ["uniform", "one turn", "file"]:
            raise ValueError("Your kicker type is not recognized.")

        if isinstance(self.B, (int, float, str)):
            pass
        elif isinstance(self.B, (np.int64, np.float64)):
            self.B = self.B.items()
        else:
            raise TypeError(f"Your kicker field (parameter 'B') is not valid.")
            

        if not isinstance(self.b_norm, (int, float, np.int64, np.float64)) and self.b_norm is not None:
            raise TypeError("Your kicker field normalization is not valid.")
        
        if not isinstance(self.t_norm, (int, float, np.int64, np.float64)):
            raise TypeError("Your kicker timing normalization is not valid.")
        
        # Warn the user in the case of unused field normalizations.
        if self.mode in ["uniform", "one turn"] and (self.b_norm is not None or self.t_norm != 0):
            warnings.warn(f"As you selected a {self.mode} kicker field, your field and time normalizations will be ignored.", SyntaxWarning, stacklevel = 2)

        if isinstance(self.kicker_num, (int, np.int64, float, np.float64)):
            self.kick_max = int(self.kick_max)
        if not self.kicker_num in [1, 3]:
            raise ValueError(f"Your kick number {self.kicker_num} is not valid.")

        if isinstance(self.kick_max, (int, np.int64, float, np.float64)):
            self.kick_max = int(self.kick_max)
        else:
            raise TypeError("Your maximum number of kicks must be an integer!")
        
        if self.mode == "uniform":
            if isinstance(self.B, (int, float)):
                t_i, t_f = -1, 1
                self.t_list = np.array([t_i, t_f], dtype = np.float64)
                self.b_k = np.array([self.B / 10000, self.B / 10000], dtype = np.float64)
            else:
                raise TypeError("For a uniform kicker, 'B' must be int or float!")
            self.kick_max = 100
            
        elif self.mode == "one turn":
            if isinstance(self.B, (int, float)):
                self.t_list = np.array([-1, 1.5E-7, 1.5000001E-7], dtype = np.float64)
                self.b_k = np.array([self.B / 10000, self.B / 10000, 0], dtype = np.float64)
            else:
                raise TypeError("For a one turn kicker, 'B' must be an int or float!")
            self.kick_max = 1
            
        else:
            if isinstance(self.B, str):
                self.b_k, self.t_list = self.load_kicker_file(filename = self.B, b_norm = self.b_norm, t_norm = self.t_norm)
            else:
                raise TypeError("B must be a string path!")

    def load_kicker_file(self, filename, b_norm, t_norm):
        # Load the necessary file.
        dir = str(pathlib.Path(__file__).parent.absolute()) + "/kicker_pulses/" + filename + ".dat" #------Changed to .dat in run 2-------#
        t, b = np.loadtxt(fname = dir, skiprows = 1, usecols = (0, 1), unpack = True)
        
        # Mask unnecessary parts of the field.
        b, t = aux.mask_zeroes(b, t)

        # Normalize strength and timing if necessary.
        if b_norm is not None:
            b *= (b_norm / b.max() / 10000)
        else:
            b /= 10000
        t = (t - t_norm) / 10**9
        return b, t
    
    def get_labels(self):
        mp_vars = [(key, value) for key, value in self.kicker_params.items() if isinstance(value, (list, np.ndarray))]
        var, values = mp_vars[0][0], mp_vars[0][1]
        labels = [f"{var}={value}" for value in values]
        return len(values), labels

    def _store_parameters(self, full_path):
        with open(full_path + "/params_readable.txt", "a") as file:
            file.write("KICKER PARAMETERS:\n")
            file.write(f"Mode: {self.mode}\n")
            if self.mode == "uniform" or self.mode == "one turn":
                file.write(f"Field strength: {self.B} G\n")
            else:
                file.write(f"Field strength: {self.B}\n")
            file.write(f"Field max normalization: {self.b_norm} G\n")
            file.write(f"Field time normalization: {self.t_norm} ns\n")
            file.write(f"Maximum number of kicks: {self.kick_max}\n")
            file.write(f"Number of kicker segments: {self.kicker_num}\n\n")
        
        def _convert_to_lists():
            for key, value in self.kicker_params.items():
                self.kicker_params[key] = [value]

        def _unconvert_from_lists():
            for key, value in self.kicker_params.items():
                self.kicker_params[key] = value[0]

        _convert_to_lists()
        df = pd.DataFrame(self.kicker_params)
        _unconvert_from_lists()


        try:
            with pd.ExcelWriter(full_path + "/parameters.xlsx", engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = 'kicker', index = False)
        except FileNotFoundError:
            with pd.ExcelWriter(full_path + "/parameters.xlsx", mode = 'w') as writer:
                df.to_excel(writer, sheet_name = 'kicker', index = False)

    @classmethod
    def load(cls, dir):
        def convert_to_list(value):
            if isinstance(value, str) and value[0] == "[":
                try:
                    return [float(val.replace("'", "")) for val in value[1:-1].split(" ")]
                except ValueError:
                    return [val.replace("'", "") for val in value[1:-1].split(", ")]
            else:
                return value

        xl_path = str(pathlib.Path(__file__).parent.absolute()) + "/results/" + dir + '/parameters.xlsx'

        df = pd.read_excel(xl_path, sheet_name = "kicker", header = 0, engine = "openpyxl")
        df = df.applymap(convert_to_list)

        if isinstance(df['b_norm'][0], list):
            b_norm = df['b_norm'][0]
        elif np.isnan(df['b_norm'][0]):
            b_norm = None
        else:
            b_norm = df['b_norm'][0].item()

        if isinstance(df['B'][0], (np.float64, np.int64)):
            B = df['B'][0].item()
        else:
            B = df['B'][0]

        if isinstance(df['kicker_num'][0], (np.float64, np.int64)):
            kicker_num = df['kicker_num'][0].item()
        else:
            kicker_num = df['kicker_num'][0] 
        return cls(
            mode = df['mode'][0],
            B = B,
            b_norm = b_norm,
            t_norm = df['t_norm'][0],
            kick_max = df['kick_max'][0],
            kicker_num = kicker_num)

    def get_loops(self):
        loops_list = [len(param) for param in self.kicker_params.values() if isinstance(param, (list, np.ndarray))]
        if len(loops_list) == 0:
            return None
        if len(loops_list) == 1:
            return loops_list[0]
        else:
            raise RuntimeError("more than one var in multiplex!")
  
    def generate_list(self, loops):
        kicker_params_list = copy.deepcopy(self.kicker_params)
        for key, value in self.kicker_params.items():
            kicker_params_list[key] = np.broadcast_to(value, loops)

        return kicker_params_list