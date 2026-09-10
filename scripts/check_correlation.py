import os
import sys
import numpy as np
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

from gwlab.dataset import generate_random_sample, load_real_noise


def waveform_features(h):
    a = np.abs(h)
    return np.array([
        a.max(),
        a.mean(),
        np.std(h),
        np.sqrt(np.mean(h ** 2)),
        np.percentile(a, 90),
        np.percentile(a, 99),
    ], dtype=float)


print("=== Генерируем 500 примеров ===")
noise = load_real_noise()
X, y = [], []
for _ in range(500):
    _, h, label, _, _ = generate_random_sample(real_noise=noise)
    X.append(waveform_features(h))
    y.append(label)
X = np.array(X) * 1e21  # нормализация
y = np.array(y)

print(f"Форма X: {X.shape}")

# Создаём DataFrame для удобного анализа
df = pd.DataFrame(X, columns=['max', 'mean', 'std', 'rms', 'p90', 'p99'])
df['label'] = y

print("\n=== Корреляционная матрица ===")
corr = df.corr()
print(corr.round(2))

print("\n=== Средние значения признаков по классам ===")
for col in ['max', 'mean', 'std', 'rms', 'p90', 'p99']:
    mean_0 = df[df['label'] == 0][col].mean()
    mean_1 = df[df['label'] == 1][col].mean()
    ratio = mean_1 / mean_0 if mean_0 != 0 else float('inf')
    print(f"{col:6s}: класс 0 = {mean_0:.2f}, класс 1 = {mean_1:.2f}, соотношение = {ratio:.2f}x")