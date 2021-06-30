import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

# =============================================================================
# Import data from ppg json data
# =============================================================================

ppg_json_filename = './feeds/ae10_data.json'
ppg_json_object = open(ppg_json_filename, 'r')
ppg_json = json.load(ppg_json_object)
t_raw = [point['time'] for point in ppg_json]
x_raw = [point['value'] for point in ppg_json]

# =============================================================================
# Interpolate date for even distribution
# =============================================================================
t = np.linspace(t_raw[0],t_raw[-1],num=len(t_raw))
x = np.interp(t, t_raw, x_raw, left=None, right=None, period=None)

# =============================================================================
# Freq analysis (for nerds)
# =============================================================================
Ts = t[1] - t[0]
fs = 1/Ts
n = len(x)
f = [value * (fs/n) for value in range(0,n)]
Y = np.fft.fft(x)
# Shift Freq
Y0 = np.fft.fftshift(Y)
f0 = [value * (fs/n) for value in range(int(-n/2),int(n/2))]
power0 = [value * 2/n for value in Y0]

x_noDC = [x_value - np.mean(x) for x_value in x]
Y_noDC = np.fft.fft(x_noDC)
Y0_noDC = np.fft.fftshift(Y_noDC)
power0_noDC = [abs(value)**2/n for value in Y0_noDC]
plt.plot(f0, power0_noDC)
plt.show()
plt.figure()

# =============================================================================
# Filtering signal (what we actually need)
# =============================================================================
x_filtered = scipy.signal.butter(5,)

plt.plot()