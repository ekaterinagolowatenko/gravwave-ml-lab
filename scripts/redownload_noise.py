import os
import numpy as np

print("📥 Используем уже скачанный GW150914 как источник шума...")

# Загружаем уже скачанные данные
gw_data = np.load('data/GW150914_H1.npy')
fs = 4096.0

# Берём первые 4 секунды (09:50:30–34 UTC) — там ТОЧНО нет сигнала
# (событие произошло в 09:50:45, то есть на 15-й секунде)
noise = gw_data[:int(4 * fs)]

print(f"Форма: {noise.shape}")
print(f"std сырого шума: {noise.std():.3e}")
print(f"пик: {np.max(np.abs(noise)):.3e}")

# Сохраняем как "реальный шум" для обучения
os.makedirs('data', exist_ok=True)
np.save('data/ligo_noise.npy', noise)
print("💾 Сохранено: data/ligo_noise.npy")
print("\nТеперь обучающий шум и данные GW150914 — из одного источника!")