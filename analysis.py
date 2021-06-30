import json
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
import cv2

def filter(t_raw, x_raw):
    """Interpolate to equal sizes, highpass filter the raw data"""
    # ==========================================================================
    # Interpolate date for even distribution
    # ==========================================================================
    t = np.linspace(t_raw[0],t_raw[-1],num=len(t_raw))
    x = np.interp(t, t_raw, x_raw, left=None, right=None, period=None)

    # ==========================================================================
    # Freq analysis (for nerds)
    # ==========================================================================
    Ts = t[1] - t[0]
    fs = 1/Ts * 1000
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
    if False:
        plt.plot(f0, power0_noDC)
        plt.show()
        plt.figure()

    # ==========================================================================
    # Filtering signal (what we actually need)
    # ==========================================================================
    lowcutBPM = 40 # Lowcut in BPM
    lowcut = lowcutBPM/60 # Cut off frequency in Hz
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = scipy.signal.butter(N=5, Wn=low, btype='highpass')
    x_filtered = scipy.signal.lfilter(b, a, x)
    # Move the line upward so it's easier to see
    #x_filtered = [x_value + x[0] for x_value in x_filtered]
    x_filtered_max = max(x_filtered)
    x_filtered = [x_value/x_filtered_max for x_value in x_filtered]
    tmin = t[0]
    # Set first value to 0 time and convert to seconds
    t = [(t_value - tmin)/1000 for t_value in t]
    skip = 30
    return(t, x, x_filtered)



def ppg_filter():
    # ==========================================================================
    # Import data from ppg json data
    # ==========================================================================
    ppg_json_filename = './feeds/ae10_data.json'
    ppg_json_object = open(ppg_json_filename, 'r')
    ppg_json = json.load(ppg_json_object)
    t_raw = [point['time'] for point in ppg_json]
    x_raw = [point['value'] for point in ppg_json]
    return(filter(t_raw, x_raw))

def spg_filter():
    # ==========================================================================
    # Import video data
    # ==========================================================================
    spg_video_filename = './feeds/ae10_video.mp4'
    spg_video = cv2.VideoCapture(spg_video_filename)
    t_raw = []
    x_raw = []

    # ==========================================================================
    # Generate x_raw and t_raw. We only need the red frames, since they serve
    # for the best purpose.
    # ==========================================================================
    previous_time = -1
    while(spg_video.isOpened()):
        ret, frame = spg_video.read()
        if ret:
            time = spg_video.get(cv2.CAP_PROP_POS_MSEC)
            if previous_time >= time:
                break
            previous_time = time
            t_raw.append(time)
            red_frame = frame[:][:][2]
            red_frame_mean = np.mean(red_frame)
            x_raw.append(red_frame_mean)
        else:
            break
    spg_video.release() 
    cv2.destroyAllWindows()

    return(filter(t_raw, x_raw))

if __name__ == '__main__':
    spg_t, spg_x,  spg_x_filtered = spg_filter()
    ppg_t, ppg_x,  ppg_x_filtered = ppg_filter()

    plt.plot(spg_t, spg_x_filtered, label="SPG")
    plt.plot(ppg_t, ppg_x_filtered, label="PPG")
    plt.legend()
    plt.ylabel("Normalized Amplitude")
    plt.xlabel("Seconds")
    plt.show()
