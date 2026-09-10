import os
import numpy as np

from gwlab import noise
from .waveforms import generate_chirp

# Путь к файлу с шумом: от текущей папки (src/gwlab) два уровня вверх, потом в data
NOISE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligo_noise.npy')

def load_real_noise():
    """Загружает сохраненный шум LIGO, если файл существует."""
    if os.path.exists(NOISE_FILE):
        return np.load(NOISE_FILE)
    return None

def generate_random_sample(fs=4096, duration=2.0, min_mass=10.0, max_mass=50.0,
                           real_noise=None):
    """
    Генерирует один пример: сигнал+шум (label=1) или просто шум (label=0).
    real_noise: массив реалистичного шума (если None — обычный гауссов).
    """
    has_signal = np.random.choice([True, False])
    n_samples = int(fs * duration)

    # Random crop: вырезаем случайный кусок шума нужной длины
    if real_noise is not None and len(real_noise) > n_samples:
        start = np.random.randint(0, len(real_noise) - n_samples)
        noise = real_noise[start:start + n_samples]
    else:
        noise = np.random.normal(0, 1e-22, n_samples)

    # Случайная "громкость" шума — чтобы задача была разной сложности
    noise = noise * np.random.uniform(0.5, 1.5)

    t = np.linspace(0, duration, n_samples)

    if has_signal:
        m1 = np.random.uniform(min_mass, max_mass)
        m2 = np.random.uniform(min_mass, max_mass)

        _, clean_h = generate_chirp(m1, m2, fs, duration)

        # Усиливаем сигнал так, чтобы он был заметно выше шума.
        clean_h = clean_h / np.max(np.abs(clean_h))
        signal_strength = np.random.uniform(50.0, 150.0)
        clean_h = clean_h * signal_strength * 1e-21

        # Добавляем сигнал в случайное место по времени.
        shift = np.random.randint(0, max(1, len(noise) - len(clean_h)))
        noisy_h = noise.copy()
        noisy_h[shift:shift + len(clean_h)] += clean_h
        label = 1
        true_m1 = m1
        true_m2 = m2
    else:
        noisy_h = noise
        label = 0
        true_m1 = 0.0
        true_m2 = 0.0

    return t, noisy_h, label, true_m1, true_m2