import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
早期草稿版本
'''
def stationary_convolution(
    r_time: np.ndarray,
    W: np.ndarray,


):
    """平稳子波合成"""
    r_time = r_time.ravel()
    W = W.ravel()


    r_time_len = len(r_time)
    W_len = len(W)
    
    time = np.arange(r_time_len)

    s_syn = np.convolve(W, r_time, mode="same")
    figure = plt.figure()
    plt.plot(time, s_syn)
    plt.show()
    return  s_syn
def time_varying_wavelet_inverison():
    pass


def nonstationary_convolution(r_time, W, twt):
    r_time = r_time.ravel()
    W = W.ravel()
    twt = twt.ravel()
    pass


if __name__ == "__main__":
    
    r_time = np.load(r"D:\python_code\python_project\seismic_well\true_data_tying\data\inversion_results\r_time_init.npy")
    W = np.load(r"D:\python_code\python_project\seismic_well\true_data_tying\data\inversion_results\w_init.npy")
    s_syn = stationary_convolution(r_time, W )
    
    