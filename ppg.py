import matplotlib.pyplot as plt
import datetime
import json
import scipy.signal as signal
import numpy as np
from scipy.signal import freqz



def parse_data(data):
    x = []
    y = []
    for point in data:
        y.append(point['value'])
        time = point['time']['$date']
        temp_x = datetime.datetime.fromtimestamp(time/1000.0)
        x.append(time)
    return(x,y)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, low, btype='highpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


ae10_file = open('./feeds/ae10_data.json','r')
ae10_data = json.load(ae10_file)

# =============================================================================
# Parsing Data here from JSON to t and x and interpolatin. Note this keeps the
# time in timestamp form (milliseconds)
# =============================================================================
t, x = parse_data(ae10_data)
tlin = np.linspace(t[0],t[-1],num=len(t))
xinterp = np.interp(tlin, t, x, left=None, right=None, period=None)
xmax = max(xinterp) + 1
xnormal = [x_value/xmax for x_value in xinterp]
# Convert time to datetime
ttime = [datetime.datetime.fromtimestamp(time/1000.0) for time in tlin]
for index in range(len(tlin)-1):
    print(tlin[index+1] - tlin[index])
# Sample rate and desired cutoff frequencies (in Hz).
fs = 5000.0
lowcut = 500.0
highcut = 1250.0
fs = 40.13 * 10**3
lowcut = 50/60

# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [3, 6, 9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order) 
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
            '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

# Filter a noisy signal.
T = 0.05
nsamples = int(T * fs)
#t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 600.0
#x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
#x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
#x += a * np.cos(2 * np.pi * f0 * t + .11)
#x += 0.03 * np.cos(2 * np.pi * 2000 * t)
plt.figure(2)
plt.clf()
plt.plot(tlin, xinterp, label='Noisy signal')

y = butter_bandpass_filter(xinterp, lowcut, highcut, fs, order=6)
plt.plot(tlin, y, label='Filtered signal (%g Hz)' % f0)
#plt.xlabel('time (seconds)')
#plt.hlines([-a, a], 0, T, linestyles='--')
#plt.grid(True)
#plt.axis('tight')
plt.legend(loc='upper left')

plt.show()
