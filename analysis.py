import json
from os import W_OK
from re import X
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
import cv2

def interpolate(t_raw, x_raw):
    # ==========================================================================
    # Interpolate date for even distribution
    # ==========================================================================
    t = np.linspace(t_raw[0],t_raw[-1],num=len(t_raw))
    x = np.interp(t, t_raw, x_raw, left=None, right=None, period=None)
    return(t,x)


def filter(t, x):
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
    return(t, x_filtered)


def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    response = [(x-data_min)/(data_max-data_min) for x in data]
    return(response)


def sync(t, x, start_time, skip=140):
    tmin = t[0]
    # Set first value to 0 time and convert to seconds
    t = [(t_value - tmin + start_time)/1000 for t_value in t]
    return(t[skip:], x[skip:])

def get_k(x):
    try:
        y = x.compressed()  # if masked array
    except AttributeError:
        y = x               # not a masked array

    ave = np.mean(y)
    std = np.std(y)
    return(std/ave)


def spg_algorithm(frame): 
    red_frame = frame[:][:][2]
    red_frame_mean = np.mean(red_frame)
    k = get_k(red_frame)
    fps = 30 #ish
    t = 1/fps
    spg =1/(2 * t * k**2)
    return(spg)


def ppg_filter():
    # ==========================================================================
    # Import data from ppg json data
    # ==========================================================================
    ppg_json_filename = './feeds/ae10_data.json'
    ppg_json_object = open(ppg_json_filename, 'r')
    ppg_json = json.load(ppg_json_object)
    t_raw = [point['time'] for point in ppg_json]
    x_raw = [point['value'] for point in ppg_json]
    start_time = 1624496030263
    t, x = interpolate(t_raw, x_raw)
    # =========================================================================
    # Research based algorithm
    # =========================================================================
    x = [1/math.log(i) for i in x]
    t, x_filtered = filter(t, x)
    t_synced, x_filtered_synced = sync(t, x_filtered, start_time)
    t, x_synced = sync(t, x, start_time)
    x_synced = normalize(x_synced)
    x_filtered_synced = normalize(x_filtered_synced)
    return(t_synced, x_synced, x_filtered_synced)

def spg_filter():
    # ==========================================================================
    # Import video data
    # ==========================================================================
    spg_video_filename = './feeds/ae10_video.mp4'
    spg_video_filename = './feeds/spg_pxl.mp4'
    spg_video = cv2.VideoCapture(spg_video_filename)
    t_raw = []
    x_raw = []

    # ==========================================================================
    # Generate x_raw and t_raw. We only need the red frames, since they serve
    # for the best purpose.
    # time for s: 17:51:02.000000000
    # ==========================================================================
    previous_time = -1
    duration = 0
    while(spg_video.isOpened()):
        ret, frame = spg_video.read()
        if ret:
            time = spg_video.get(cv2.CAP_PROP_POS_MSEC)
            if previous_time >= time:
                break
            previous_time = time
            t_raw.append(time)
            intensity = spg_algorithm(frame)
            x_raw.append(intensity)
            duration = spg_video.get(cv2.CAP_PROP_POS_MSEC)
        else:
            break
    spg_video.release() 
    cv2.destroyAllWindows()
    print(duration)
    start_time = 1624496044000 - duration
    # =========================================================================
    # Research based algorithm 
    # =========================================================================
    t, x = interpolate(t_raw, x_raw)
    t, x_filtered = filter(t, x)
    t_synced, x_filtered_synced = sync(t, x_filtered, start_time)
    t, x_synced = sync(t, x, start_time)
    x_synced = normalize(x_synced)
    x_filtered_synced = normalize(x_filtered_synced)
    return(t_synced, x_synced, x_filtered_synced)

if __name__ == '__main__':
    spg_t, spg_x,  spg_x_filtered = spg_filter()
    # ppg_t, ppg_x,  ppg_x_filtered = ppg_filter()
    print(len(spg_t), len(spg_x),  len(spg_x_filtered))

    plt.plot(spg_t, spg_x_filtered, label="SPG")
    # plt.plot(ppg_t, ppg_x_filtered, label="PPG")
    plt.legend()
    plt.ylabel("Normalized Amplitude")
    plt.xlabel("Seconds")
    plt.show()
