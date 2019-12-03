import numpy as np
from scipy import signal

class Signal_processing:
    def __init__(self, buffer_length):
        self.BUFFER_LENGTH = buffer_length

    def normalization(self, buffer):
        '''
        normalize the input data buffer
        '''

        norm_data = buffer/np.linalg.norm(buffer)
        
        return norm_data

    def signal_detrending(self, buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(buffer)
        
        return detrended_data

    def interpolation(self, data_buffer, times):
        '''
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        '''
        
        even_times = np.linspace(times[0], times[-1], self.BUFFER_LENGTH)
        
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):    
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = signal.lfilter(b, a, data_buffer)
        
        return filtered_data

    def fft(self, data_buffer, fps):
        freqs = float(fps) / self.BUFFER_LENGTH * np.arange(self.BUFFER_LENGTH / 2 + 1)
        # print(freqs)
        freqs_in_minute = 60. * freqs
        # print(freqs_in_minute)
        print(data_buffer)
        
        raw_fft = np.fft.rfft(data_buffer*30)
        fft = np.abs(raw_fft)**2
        print(fft)
        
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        interest_idx_sub = interest_idx[:-1].copy() #advoid the indexing error
        freqs_of_interest = freqs_in_minute[interest_idx_sub]
        
        fft_of_interest = fft[interest_idx_sub]
        # print(fft_of_interest)
        print(freqs_of_interest)
        
        return fft_of_interest, freqs_of_interest

from matplotlib import pyplot as plt
if __name__ == "__main__":
    fps = 61
    sp = Signal_processing(buffer_length=fps*5)
    with open("hrv.data") as f:
        data = f.readlines()

    data = np.array([float(v) for v in data])
    # print(len(data))

    norm = sp.normalization(data)
    # plt.plot(norm)
    # plt.show()

    detrend = sp.signal_detrending(norm)
    # plt.plot(detrend)
    # plt.show()

    # bpf = sp.butter_bandpass_filter(detrend,0.8,4,fps)
    # plt.plot(bpf)
    # plt.show()

    fft,freq = sp.fft(detrend, fps)
    print(len(fft))
    plt.plot(freq,fft)
    plt.show()
