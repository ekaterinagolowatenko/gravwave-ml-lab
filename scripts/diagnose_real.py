import os
import sys
import numpy as np
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from gwlab.features import extract_features
from gwlab.dataset import load_real_noise

bundle = joblib.load(os.path.join(project_root, 'models', 'rf_baseline.joblib'))
model, scaler = bundle['model'], bundle['scaler']

train_noise = load_real_noise()
h = np.load(os.path.join(project_root, 'data', 'GW150914_H1.npy'))
fs = 4096.0
seg_real = h[int(10 * fs):int(12 * fs)]   # 09:50:40–42: сигнала ещё НЕТ

print("\n=== Сырые статистики (окно 2 с) ===")
for name, a in [("train noise 12.09", train_noise[:8192]), ("GW150914 pre", seg_real)]:
    print(f"{name:18s} std={np.std(a):.3e}  max={np.max(np.abs(a)):.3e}")

names = ['max', 'mean', 'std', 'rms', 'p90', 'p99', 'e_low', 'e_mid', 'e_high', 'ratio', 'rise', 'peakpos']
f_train = extract_features(train_noise[:8192])
f_real = extract_features(seg_real)

print("\n=== Признаки ДО scaler ===")
print(f"{'признак':8s}{'train noise':>14s}{'GW pre':>14s}")
for n, a, b in zip(names, f_train, f_real):
    print(f"{n:8s}{a:14.3e}{b:14.3e}")

print("\n=== После scaler и вердикт модели ===")
for name, f in [("train noise", f_train), ("GW pre", f_real)]:
    z = scaler.transform(f.reshape(1, -1))[0]
    p = model.predict_proba(z.reshape(1, -1))[0][1]
    print(f"{name:12s} min z={z.min():7.1f}  max z={z.max():7.1f}  → P(чирп)={p:.3f}")