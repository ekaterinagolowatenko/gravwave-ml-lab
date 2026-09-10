import numpy as np

def add_white_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise