import numpy as np
from scipy import signal


def extract_features(h, fs=4096):
    """
    Извлекает разнообразные признаки из сигнала.
    Возвращает массив из ~20 признаков.
    """
    a = np.abs(h)
    
    # === АМПЛИТУДНЫЕ ПРИЗНАКИ ===
    amp_features = [
        a.max(),
        a.mean(),
        np.std(h),
        np.sqrt(np.mean(h ** 2)),
        np.percentile(a, 90),
        np.percentile(a, 99),
    ]
    
    # === ЧАСТОТНЫЕ ПРИЗНАКИ (FFT) ===
    fft_vals = np.fft.rfft(h)
    freqs = np.fft.rfftfreq(len(h), d=1.0/fs)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # Энергия в разных частотных диапазонах
    low_freq_mask = (freqs >= 20) & (freqs < 100)
    mid_freq_mask = (freqs >= 100) & (freqs < 500)
    high_freq_mask = (freqs >= 500)
    
    energy_low = np.sum(power_spectrum[low_freq_mask])
    energy_mid = np.sum(power_spectrum[mid_freq_mask])
    energy_high = np.sum(power_spectrum[high_freq_mask])
    
    freq_features = [
        energy_low,
        energy_mid,
        energy_high,
        energy_low / (energy_mid + energy_high + 1e-10),  # соотношение
    ]
    
    # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
    # Скорость нарастания амплитуды (envelope)
    envelope = np.abs(signal.hilbert(h))
    rise_rate = np.max(envelope[len(envelope)//2:]) / (np.mean(envelope[:len(envelope)//2]) + 1e-10)
    
    time_features = [
        rise_rate,
        np.argmax(envelope) / len(envelope),  # позиция пика (нормализованная)
    ]
    
    # Объединяем все признаки
    all_features = amp_features + freq_features + time_features
    
    return np.array(all_features, dtype=float)