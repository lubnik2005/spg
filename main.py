# #############################################################################
# Author: Nik
# Description: Converts video of finger into heart bpm
# #############################################################################

# =============================================================================
# Values to experiment with
# =============================================================================
# Butterworth
from typing import List, Set, Dict, Tuple, Optional
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import normalize
import datetime
import math
import numpy as np
import cv2
import os
import pandas as pd
import os
ORDER: int = 2      # Order of the butterworth filter
BPM_L: int = 50     # [bpm] The lowest heart rate
BPM_H: int = 100    # [bpm] The hightest heart rate
# [s] Amount of time to cut from the beggining while fiter is stabilizing
FILTER_STABILIZATION_TIME: int = 5
WINDOW_SECONDS:            int = 6      # [s] Sliding window duration
BPM_SAMPLING_PERIOD:       int = 0.5    # [s] Time between heart rate estimations
FINE_TUNING_FREQ_INCREMENT:int = 1      # [bpm] Separation between test tones for smoothing



# =============================================================================
# Imports
# =============================================================================


# =============================================================================
# Retreives the video and makes sure that it exists and is readable
# =============================================================================

def acquire() -> bool:
    while True:
        feed_name: str = input('Please input feed name: ')
        video_file_path: str = os.path.join(
            os.getcwd(), 'feeds', feed_name + '.mp4')
        ekg_file_path: str = os.path.join(
            os.getcwd(), 'feeds', feed_name + '.txt')
        if not (os.path.isfile(video_file_path) and os.path.isfile(ekg_file_path)):
            print("Was not able to find files:")
            print("    1.", video_file_path)
            print("    2.", ekg_file_path)
        else:
            video: VideoCapture = cv2.VideoCapture(video_file_path)
            ekg = "TEMP"

            if not (video.isOpened()):
                print("Error opening video stream or file")
            else:
                print("Successfully imported files")
                break

    return(video, ekg)

# =============================================================================
# Used to view the video. Not for production
# =============================================================================


def view_video(video):

    # Read until video is completed
    while(video.isOpened()):
        # videoture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                # Break the loop
        else:
            break
    # When everything done, release the video videoture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# =============================================================================
# Given a VideoCapture object returns an array of red pixel averages.
# =============================================================================


def get_red_pixels(video) -> List[float]:
    brightness: List[float] = []
    # Read until video is completed
    while(video.isOpened()):
        # videoture frame-by-frame
        ret, frame = video.read()

        if ret == True:
            red_frame = frame[:][:][2]
            red_frame_mean = np.mean(red_frame)
            brightness.append(red_frame_mean)
        else:
            break
    # When everything done, release the video videoture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return(brightness)

# =============================================================================
# Given signal and sample frequency, deterine the filtered signal
# =============================================================================


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# =============================================================================
# Remove the initial stabilization time of the butterworth filter.
# =============================================================================


def remove_stabilization(fbrightness, fs, seconds):
    initial_time = round(fs * seconds)
    fbrightness = fbrightness[initial_time:-1]
    return(fbrightness)
# =============================================================================
# Converts the signal to freq domain
# =============================================================================


def convert_to_freq(fbrightness):
    fbrightness = np.array(fbrightness)
    fftMagnitude = np.fft.fft(fbrightness)
    fftShiftMagnitude = np.fft.fftshift(fftMagnitude)
    return(fftShiftMagnitude)


def main() -> bool:
    start = datetime.datetime.now()
    # Acuiring Video
    video, ekg = acquire()
    frames_per_second:  float = video.get(cv2.CAP_PROP_FPS)
    brightness = get_red_pixels(video)
    np.savetxt('data.csv', np.array(brightness), delimiter=",")
    #plt.plot(brightness)
    #plt.show()

    # Cutoff Frequencies
    fcl = BPM_L / 60
    fch = BPM_H / 60

    # Filtering
    fbrightness = butter_bandpass_filter(
        brightness, fcl, fch, frames_per_second, order=ORDER)
    fbrightness = remove_stabilization(
        fbrightness, frames_per_second, FILTER_STABILIZATION_TIME)
    
    plt.plot(fbrightness)
    plt.show()

    # Precalculations
    num_window_samples = round(WINDOW_SECONDS * frames_per_second)
    bpm_sampling_period_samples = round(BPM_SAMPLING_PERIOD * frames_per_second)
    num_bpm_samples = math.floor((len(fbrightness) - num_window_samples) / bpm_sampling_period_samples)
    bpm = [None] * num_bpm_samples
    bpm_smooth = [None] * num_bpm_samples

    # Create Graph
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    # Starts on 1 becuase of chuffing MATLAB!!!!!
    for i in range(num_bpm_samples):
        # Fill sliding window with original signal values
        window_start = (i) * bpm_sampling_period_samples
        fbrightness_window = fbrightness[window_start:window_start +
                                         num_window_samples]
        # Use Hanning window to bring edges to zero. In this way, no artificial
        # high frequencies appear when the signal is treated as periodic by the FFT
        fbrightness_hann = fbrightness_window * \
            np.transpose(np.hanning(len(fbrightness_window)))

        gain = np.abs(np.fft.fft(fbrightness_hann))

        # FFT indices of frequencies where the human heartbeat is
        il = math.floor(
            fcl * (len(fbrightness_window) / frames_per_second))
        ih = math.ceil(fch * (len(fbrightness_window) / frames_per_second))
        index_range = np.array(range(il, ih))
        # print(len(index_range))
        # print(len(gain[index_range]))

        #plt.plot(index_range * (frames_per_second / len(fbrightness_hann)) * 60, gain[index_range])
        #plt.show()

        pks, pks_props=signal.find_peaks(gain[index_range])
        #if len(pks) == 0:
        #
        #    continue
        [max_peak_v, max_peak_i]=pks.max(0), pks.argmax(0)
        max_f_index = index_range[pks[max_peak_i]]
        bpm_value   = (max_f_index) * (frames_per_second / len(fbrightness_hann)) * 60 
        bpm[i] = (bpm_value)

        # Smooth the highest peak frequency by finding the frequency that
        # best "correlates" in the resolution range around the peak
        freq_resolution = 1 / WINDOW_SECONDS
        lowf = bpm_value / 60 - 0.5 * freq_resolution
        freq_inc = FINE_TUNING_FREQ_INCREMENT / 60
        test_freqs = round(freq_resolution / freq_inc)
        freqs = [(i * freq_inc + lowf) for i in list(range(test_freqs))]
        power = np.array([0]) # The first item is none because this is a copy of stupid Matlab code
        for h in range(test_freqs):
            re = 0
            im = 0
            for j in range(0, (len(fbrightness_hann))):
                phi = 2 * math.pi * freqs[h] * (j / frames_per_second)
                re = re + fbrightness_hann[j] * math.cos(phi)
                im = im + fbrightness_hann[j] * math.sin(phi)
            #power[h] = re * re + im * im
            power = np.append(power, re * re + im * im)
        [max_peak_v, max_peak_i] = power.max(0), power.argmax(0)-1
        bpm_smooth_value = 60*freqs[max_peak_i]
        bpm_smooth[i] = bpm_smooth_value
        print(bpm)

    plt.plot(bpm_smooth)
    plt.show()


    end=datetime.datetime.now()
    print(end - start)
    return(True)


main()
# =============================================================================
# Import Data from EKG/PPG "AFE4950"
# =============================================================================
# FIXME: This is being imported incorrectly. The first column is *not* an indexing
"""
afe_file = os.path.join('test_data', 'AFE4950_CAPTURED_DATA.csv')
df = pd.read_csv(afe_file, index_col=False)
ECG = butter_bandpass_filter(df['ECG_Value'], 40, 100, 500, order=5)
plt.plot(df['ECG_Time'][0:1000], ECG[0:1000]/700)
#plt.plot(df['TIA1-3_Time'], df['TIA1-3_Value'] - np.mean(df['TIA1-3_Value']))
plt.show()
print(df.head())
"""

