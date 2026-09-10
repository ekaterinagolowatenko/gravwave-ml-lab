import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

from gwlab.dataset import generate_random_sample, load_real_noise

print("=== Генерируем 500 примеров ===")
noise = load_real_noise()
X_raw, y = [], []
for _ in range(500):
    _, h, label, _, _ = generate_random_sample(real_noise=noise)
    X_raw.append(h)
    y.append(label)
X_raw = np.array(X_raw)
y = np.array(y)

print(f"Всего примеров: {len(y)}")
print(f"Класс 0 (шум): {np.sum(y == 0)}")
print(f"Класс 1 (сигнал): {np.sum(y == 1)}")

print("\n=== Проверяем пики амплитуды ===")
peaks = np.max(np.abs(X_raw), axis=1)
print(f"Класс 0: среднее {peaks[y == 0].mean():.3e}, min {peaks[y == 0].min():.3e}, max {peaks[y == 0].max():.3e}")
print(f"Класс 1: среднее {peaks[y == 1].mean():.3e}, min {peaks[y == 1].min():.3e}, max {peaks[y == 1].max():.3e}")

print("\n=== Простейший пороговый классификатор ===")
threshold = (peaks[y == 0].mean() + peaks[y == 1].mean()) / 2
print(f"Порог: {threshold:.3e}")
pred_simple = (peaks > threshold).astype(int)
acc_simple = accuracy_score(y, pred_simple)
print(f"Точность порогового классификатора: {acc_simple * 100:.1f}%")

print("\n=== Извлекаем признаки ===")
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

X_feat = np.vstack([waveform_features(h) for h in X_raw])
print(f"Форма X_feat: {X_feat.shape}")
print(f"Первый пример (признаки): {X_feat[0]}")
print(f"Первый пример (label): {y[0]}")

print("\n=== Обучаем Random Forest на признаках ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Train class 0: {np.sum(y_train == 0)}, class 1: {np.sum(y_train == 1)}")
print(f"Test class 0: {np.sum(y_test == 0)}, class 1: {np.sum(y_test == 1)}")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred_rf = model.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)
print(f"\n🎯 Random Forest Accuracy: {acc_rf * 100:.1f}%")
print(f"Предсказания RF: класс 0 = {np.sum(pred_rf == 0)}, класс 1 = {np.sum(pred_rf == 1)}")

print("\n=== Обучаем Random Forest на сырых данных (8192 точки) ===")
X_train_raw, X_test_raw, _, _ = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
model_raw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_raw.fit(X_train_raw, y_train)
pred_raw = model_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test, pred_raw)
print(f"🎯 Raw Data Accuracy: {acc_raw * 100:.1f}%")
print(f"Предсказания Raw: класс 0 = {np.sum(pred_raw == 0)}, класс 1 = {np.sum(pred_raw == 1)}")