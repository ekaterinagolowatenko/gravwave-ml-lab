import numpy as np

def add_white_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def bandpass(h, fs=4096.0, f_low=20.0, f_high=300.0):
    """ОДИН фильтр для обучения и детекции: оставляем полосу чирпа."""
    from scipy import signal as sps
    b, a = sps.butter(4, [f_low, f_high], btype='band', fs=fs)
    return sps.filtfilt(b, a, h)