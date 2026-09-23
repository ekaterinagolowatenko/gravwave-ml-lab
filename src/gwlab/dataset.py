import os
import numpy as np
from .waveforms import generate_chirp

# Путь к файлу с шумом: от текущей папки (src/gwlab) два уровня вверх, потом в data
NOISE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligo_noise.npy')


def load_real_noise():
    """Загружает сохраненный шум LIGO, если файл существует."""
    if os.path.exists(NOISE_FILE):
        noise = np.load(NOISE_FILE)
        print(f"✅ Используется РЕАЛЬНЫЙ шум LIGO: {NOISE_FILE}")
        print(f"   Форма: {noise.shape}, std: {noise.std():.3e}, "
              f"min: {noise.min():.3e}, max: {noise.max():.3e}")
        return noise
    print(f"⚠️  Файл {NOISE_FILE} не найден — используется СИНТЕТИЧЕСКИЙ "
          f"гауссов шум (np.random.normal(0, 1e-22, ...)).")
    print(f"   Если хочешь реальный шум LIGO, запусти scripts/download_ligo_noise.py")
    return None


def generate_random_sample(fs=4096, duration=2.0, min_mass=10.0, max_mass=50.0,
                           real_noise=None, snr_range=(0.5, 30.0)):
    """
    Генерирует один пример: сигнал+шум (label=1) или просто шум (label=0).
    real_noise: массив реалистичного шума (если None — обычный гауссов).
    snr_range: диапазон значений SNR для регулировки отношения сигнала к шуму.
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
        # 1. СНАЧАЛА массы
        m1 = np.random.uniform(min_mass, max_mass)
        m2 = np.random.uniform(min_mass, max_mass)

        # 2. Чирп КОРОЧЕ окна: 1 секунда, слияние в конце этого отрезка
        chirp_duration = 0.25   # как у реального чирпа в полосе детектора
        _, clean_h = generate_chirp(m1, m2, fs, chirp_duration)

        # 3. Нормализация пика к 1
        clean_h = clean_h / np.max(np.abs(clean_h))

        # 4-5. Масштаб относительно шума: лог-SNR
        noise_std = np.std(noise)
        snr = 10 ** np.random.uniform(np.log10(snr_range[0]), np.log10(snr_range[1]))
        clean_h = clean_h * snr * noise_std

        # 6-8. Вклейка в случайное место: shift живой (4096 позиций)
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