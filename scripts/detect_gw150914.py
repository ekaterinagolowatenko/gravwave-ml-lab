import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
from gwlab.features import extract_features, extract_wavelet_features   # 2. ПОТОМ импорт gwlab


# Абсолютные пути — работают из ЛЮБОЙ папки
model_path = os.path.join(project_root, 'models', 'rf_baseline.joblib')
data_path  = os.path.join(project_root, 'data', 'GW150914_H1.npy')
meta_path  = os.path.join(project_root, 'data', 'GW150914_H1_meta.npy')
out_path   = os.path.join(project_root, 'docs', 'gw150914_detection.png')
# Загружаем модель (ОДИН раз!)
print(f"📦 Загружаем модель: {model_path}")
bundle = joblib.load(model_path)
model, scaler = bundle['model'], bundle['scaler']

# Загружаем данные
print(f"📡 Читаем GW150914: {data_path}")
h = np.load(data_path)
meta = np.load(meta_path, allow_pickle=True).item()
fs = meta['sample_rate']
print(f"Форма: {h.shape}, частота: {fs} Гц")

from gwlab.noise import bandpass
h = bandpass(h, fs=fs)
print(f"После band-pass: std={h.std():.3e}")

# Обрезаем до нужных 10 секунд
start_offset = int(10 * fs)
h = h[start_offset:start_offset + int(10 * fs)]

# Скользящее окно
window = int(2 * fs)
step = int(0.25 * fs)

print("🔍 Прогоняем скользящее окно...")
times, probs = [], []
for start in range(0, len(h) - window, step):
    seg = h[start:start + window]
    feats = np.concatenate([extract_features(seg), extract_wavelet_features(seg)]).reshape(1, -1) #склеивает два массива в один: 12 + 16 = 28 признаков — ровно то, на чём обучалась RF+W.
    feats = scaler.transform(feats)
    p = model.predict_proba(feats)[0][1]
    times.append(start / fs)
    probs.append(p)

# График
plt.figure(figsize=(10, 4))
plt.plot(times, probs, marker='o', linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', label='порог решения')
plt.axvline(x=5, color='green', linestyle=':', alpha=0.7, label='момент слияния (t≈5c)')
plt.xlabel('секунды от 09:50:40 UTC')
plt.ylabel('вероятность «это чирп!»')
plt.title('GW150914: модель на синтетике узнаёт НАСТОЯЩУЮ волну!')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"💾 График: {out_path}")

# Финальный отчёт
best_prob = max(probs)
best_t = times[probs.index(best_prob)]
print(f"\n🎯 МАКСИМАЛЬНАЯ ВЕРОЯТНОСТЬ: {best_prob:.3f} в момент t = {best_t:.2f} c")
if best_prob > 0.9 and 3 < best_t < 7:
    print(" Модель обнаружила гравитационную волну в правильном месте!")