import os
import numpy as np
import warnings
from astropy import constants as const, units as u

warnings.filterwarnings('ignore')

dire_base = os.getenv("DIRE_BASE")

def freq_to_x(frequency):
    frequency *= u.GHz
    return (const.h * frequency.to(u.s**-1) / (const.k_B * 2.725 * u.K)).value

def load_passband_frequency(fname, dire_base):
    return np.load(f"{dire_base}/passbands/{fname}")

def load_passband_transmission(fname, dire_base):
    return np.load(f"{dire_base}/passbands/{fname}")

def integrate_passband(bandpass, freq_arr):
    del_freq = ((np.max(freq_arr) - np.min(freq_arr))  / len(freq_arr))
    return np.trapz(y=np.array(bandpass), x=np.array(freq_arr), dx=del_freq)

# Load passband transmissions
passband_pa4_150 = load_passband_transmission("passband_pa4_f150.npy", dire_base=dire_base)
passband_pa4_220 = load_passband_transmission("passband_pa4_f220.npy", dire_base=dire_base)
passband_pa5_098 = load_passband_transmission("passband_pa5_f098.npy", dire_base=dire_base)
passband_pa5_150 = load_passband_transmission("passband_pa5_f150.npy", dire_base=dire_base)
passband_pa6_098 = load_passband_transmission("passband_pa6_f098.npy", dire_base=dire_base)
passband_pa6_150 = load_passband_transmission("passband_pa6_f150.npy", dire_base=dire_base)

passband_npipe_030 = load_passband_transmission("passband_npipe_f030.npy", dire_base=dire_base)
passband_npipe_044 = load_passband_transmission("passband_npipe_f044.npy", dire_base=dire_base)
passband_npipe_070 = load_passband_transmission("passband_npipe_f070.npy", dire_base=dire_base)
passband_npipe_100 = load_passband_transmission("passband_npipe_f100.npy", dire_base=dire_base)
passband_npipe_143 = load_passband_transmission("passband_npipe_f143.npy", dire_base=dire_base)
passband_npipe_217 = load_passband_transmission("passband_npipe_f217.npy", dire_base=dire_base)
passband_npipe_353 = load_passband_transmission("passband_npipe_f353.npy", dire_base=dire_base)
passband_npipe_545 = load_passband_transmission("passband_npipe_f545.npy", dire_base=dire_base) 
passband_npipe_857 = load_passband_transmission("passband_npipe_f857.npy", dire_base=dire_base)

passband_dict = {"passband_pa4_150": passband_pa4_150,
                 "passband_pa4_220": passband_pa4_220,
                 "passband_pa5_098": passband_pa5_098,
                 "passband_pa5_150": passband_pa5_150,
                 "passband_pa6_098": passband_pa6_098,
                 "passband_pa6_150": passband_pa6_150,

                 "passband_npipe_030": passband_npipe_030,
                 "passband_npipe_044": passband_npipe_044,
                 "passband_npipe_070": passband_npipe_070,
                 "passband_npipe_100": passband_npipe_100,
                 "passband_npipe_143": passband_npipe_143,
                 "passband_npipe_217": passband_npipe_217,
                 "passband_npipe_353": passband_npipe_353,
                 "passband_npipe_545": passband_npipe_545,
                 "passband_npipe_857": passband_npipe_857}

# Load passband frequencies
freq_pa4_150 = load_passband_frequency("freq_pa4_f150.npy", dire_base=dire_base)
freq_pa4_220 = load_passband_frequency("freq_pa4_f220.npy", dire_base=dire_base)
freq_pa5_098 = load_passband_frequency("freq_pa5_f098.npy", dire_base=dire_base)
freq_pa5_150 = load_passband_frequency("freq_pa5_f150.npy", dire_base=dire_base)
freq_pa6_098 = load_passband_frequency("freq_pa6_f098.npy", dire_base=dire_base)
freq_pa6_150 = load_passband_frequency("freq_pa6_f150.npy", dire_base=dire_base)

freq_npipe_030 = load_passband_frequency("freq_npipe_f030.npy", dire_base=dire_base)
freq_npipe_044 = load_passband_frequency("freq_npipe_f044.npy", dire_base=dire_base)
freq_npipe_070 = load_passband_frequency("freq_npipe_f070.npy", dire_base=dire_base)
freq_npipe_100 = load_passband_frequency("freq_npipe_f100.npy", dire_base=dire_base)
freq_npipe_143 = load_passband_frequency("freq_npipe_f143.npy", dire_base=dire_base)
freq_npipe_217 = load_passband_frequency("freq_npipe_f217.npy", dire_base=dire_base)
freq_npipe_353 = load_passband_frequency("freq_npipe_f353.npy", dire_base=dire_base)
freq_npipe_545 = load_passband_frequency("freq_npipe_f545.npy", dire_base=dire_base) 
freq_npipe_857 = load_passband_frequency("freq_npipe_f857.npy", dire_base=dire_base)

freq_dict = {"freq_pa4_150": freq_pa4_150,
             "freq_pa4_220": freq_pa4_220,
             "freq_pa5_098": freq_pa5_098,
             "freq_pa5_150": freq_pa5_150,
             "freq_pa6_098": freq_pa6_098,
             "freq_pa6_150": freq_pa6_150,

             "freq_npipe_030": freq_npipe_030,
             "freq_npipe_044": freq_npipe_044,
             "freq_npipe_070": freq_npipe_070,
             "freq_npipe_100": freq_npipe_100,
             "freq_npipe_143": freq_npipe_143,
             "freq_npipe_217": freq_npipe_217,
             "freq_npipe_353": freq_npipe_353,
             "freq_npipe_545": freq_npipe_545,
             "freq_npipe_857": freq_npipe_857}


del_freq_pa4_150 = ((np.max(freq_pa4_150) - np.min(freq_pa4_150))  / len(freq_pa4_150))
del_freq_pa4_220 = ((np.max(freq_pa4_220) - np.min(freq_pa4_220))  / len(freq_pa4_220))
del_freq_pa5_098 = ((np.max(freq_pa5_098) - np.min(freq_pa5_098))  / len(freq_pa5_098))
del_freq_pa5_150 = ((np.max(freq_pa5_150) - np.min(freq_pa5_150))  / len(freq_pa5_150))
del_freq_pa6_098 = ((np.max(freq_pa6_098) - np.min(freq_pa6_098))  / len(freq_pa6_098))
del_freq_pa6_150 = ((np.max(freq_pa6_150) - np.min(freq_pa6_150))  / len(freq_pa6_150))

del_freq_npipe_030 = ((np.max(freq_npipe_030) - np.min(freq_npipe_030))  / len(freq_npipe_030))
del_freq_npipe_044 = ((np.max(freq_npipe_044) - np.min(freq_npipe_044))  / len(freq_npipe_044))
del_freq_npipe_070 = ((np.max(freq_npipe_070) - np.min(freq_npipe_070))  / len(freq_npipe_070))
del_freq_npipe_100 = ((np.max(freq_npipe_100) - np.min(freq_npipe_100))  / len(freq_npipe_100))
del_freq_npipe_143 = ((np.max(freq_npipe_143) - np.min(freq_npipe_143))  / len(freq_npipe_143))
del_freq_npipe_217 = ((np.max(freq_npipe_217) - np.min(freq_npipe_217))  / len(freq_npipe_217))
del_freq_npipe_353 = ((np.max(freq_npipe_353) - np.min(freq_npipe_353))  / len(freq_npipe_353))
del_freq_npipe_545 = ((np.max(freq_npipe_545) - np.min(freq_npipe_545))  / len(freq_npipe_545))
del_freq_npipe_857 = ((np.max(freq_npipe_857) - np.min(freq_npipe_857))  / len(freq_npipe_857))

del_freq_dict = {"del_freq_pa4_150": del_freq_pa4_150,
                 "del_freq_pa4_220": del_freq_pa4_220,
                 "del_freq_pa5_098": del_freq_pa5_098,
                 "del_freq_pa5_150": del_freq_pa5_150,
                 "del_freq_pa6_098": del_freq_pa6_098,
                 "del_freq_pa6_150": del_freq_pa6_150, 
                 "del_freq_npipe_030": del_freq_npipe_030,
                 "del_freq_npipe_044": del_freq_npipe_044,
                 "del_freq_npipe_070": del_freq_npipe_070,
                 "del_freq_npipe_100": del_freq_npipe_100,
                 "del_freq_npipe_143": del_freq_npipe_143,
                 "del_freq_npipe_217": del_freq_npipe_217,
                 "del_freq_npipe_353": del_freq_npipe_353,
                 "del_freq_npipe_545": del_freq_npipe_545,
                 "del_freq_npipe_857": del_freq_npipe_857}

# x arrays
x_pa4_150 = freq_to_x(freq_pa4_150)
x_pa4_220 = freq_to_x(freq_pa4_220)
x_pa5_098 = freq_to_x(freq_pa5_098)
x_pa5_150 = freq_to_x(freq_pa5_150)
x_pa6_098 = freq_to_x(freq_pa6_098)
x_pa6_150 = freq_to_x(freq_pa6_150)

x_npipe_030 = freq_to_x(freq_npipe_030)
x_npipe_044 = freq_to_x(freq_npipe_044)
x_npipe_070 = freq_to_x(freq_npipe_070)
x_npipe_100 = freq_to_x(freq_npipe_100)
x_npipe_143 = freq_to_x(freq_npipe_143)
x_npipe_217 = freq_to_x(freq_npipe_217)
x_npipe_353 = freq_to_x(freq_npipe_353)
x_npipe_545 = freq_to_x(freq_npipe_545) 
x_npipe_857 = freq_to_x(freq_npipe_857)

x_dict = {"x_pa4_150": x_pa4_150,
          "x_pa4_220": x_pa4_220,
          "x_pa5_098": x_pa5_098,
          "x_pa5_150": x_pa5_150,
          "x_pa6_098": x_pa6_098,
          "x_pa6_150": x_pa6_150,    
          "x_npipe_030": x_npipe_030,
          "x_npipe_044": x_npipe_044,
          "x_npipe_070": x_npipe_070,
          "x_npipe_100": x_npipe_100,
          "x_npipe_143": x_npipe_143,
          "x_npipe_217": x_npipe_217,
          "x_npipe_353": x_npipe_353,
          "x_npipe_545": x_npipe_545,
          "x_npipe_857": x_npipe_857}

# Integral over the bandpass
# ACT
passband_pa4_150_int = integrate_passband(bandpass=passband_pa4_150, freq_arr=freq_pa4_150)
passband_pa4_220_int = integrate_passband(bandpass=passband_pa4_220, freq_arr=freq_pa4_220)
passband_pa5_098_int = integrate_passband(bandpass=passband_pa5_098, freq_arr=freq_pa5_098)
passband_pa5_150_int = integrate_passband(bandpass=passband_pa5_150, freq_arr=freq_pa5_150)
passband_pa6_098_int = integrate_passband(bandpass=passband_pa6_098, freq_arr=freq_pa6_098)
passband_pa6_150_int = integrate_passband(bandpass=passband_pa6_150, freq_arr=freq_pa6_150)

# Planck
passband_npipe_030_int = integrate_passband(bandpass=passband_npipe_030, freq_arr=freq_npipe_030)
passband_npipe_044_int = integrate_passband(bandpass=passband_npipe_044, freq_arr=freq_npipe_044)
passband_npipe_070_int = integrate_passband(bandpass=passband_npipe_070, freq_arr=freq_npipe_070)
passband_npipe_100_int = integrate_passband(bandpass=passband_npipe_100, freq_arr=freq_npipe_100)
passband_npipe_143_int = integrate_passband(bandpass=passband_npipe_143, freq_arr=freq_npipe_143)
passband_npipe_217_int = integrate_passband(bandpass=passband_npipe_217, freq_arr=freq_npipe_217)
passband_npipe_353_int = integrate_passband(bandpass=passband_npipe_353, freq_arr=freq_npipe_353)
passband_npipe_545_int = integrate_passband(bandpass=passband_npipe_545, freq_arr=freq_npipe_545) 
passband_npipe_857_int = integrate_passband(bandpass=passband_npipe_857, freq_arr=freq_npipe_857)  

passband_int_dict = {"passband_pa4_150_int": passband_pa4_150_int,
                     "passband_pa4_220_int": passband_pa4_220_int,
                     "passband_pa5_098_int": passband_pa5_098_int,
                     "passband_pa5_150_int": passband_pa5_150_int,
                     "passband_pa6_098_int": passband_pa6_098_int,
                     "passband_pa6_150_int": passband_pa6_150_int,

                     "passband_npipe_030_int": passband_npipe_030_int,
                     "passband_npipe_044_int": passband_npipe_044_int,
                     "passband_npipe_070_int": passband_npipe_070_int,
                     "passband_npipe_100_int": passband_npipe_100_int,
                     "passband_npipe_143_int": passband_npipe_143_int,
                     "passband_npipe_217_int": passband_npipe_217_int,
                     "passband_npipe_353_int": passband_npipe_353_int,
                     "passband_npipe_545_int": passband_npipe_545_int,
                     "passband_npipe_857_int": passband_npipe_857_int}      

    
