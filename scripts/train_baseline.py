import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

from gwlab.dataset import generate_random_sample, load_real_noise
from gwlab.features import extract_features
from sklearn.preprocessing import StandardScaler


def waveform_features(h):
    """Простые признаки сигнала."""
    a = np.abs(h)
    return np.array([
        a.max(),
        a.mean(),
        np.std(h),
        np.sqrt(np.mean(h ** 2)),
        np.percentile(a, 90),
        np.percentile(a, 99),
    ], dtype=float)


def build_dataset(num_samples, real_noise):
    X, y = [], []
    for _ in range(num_samples):
        _, h, label, _, _ = generate_random_sample(real_noise=real_noise)
        X.append(extract_features(h))
        y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("🧠 Генерируем датасет...")
    real_noise = load_real_noise()
    X, y = build_dataset(2000, real_noise)
    print(f"Форма X: {X.shape}")

    # НОРМАЛИЗАЦИЯ (в правильном месте — после создания X!)
    X_norm = X * 1e21
    print(f"После нормализации - среднее: {X_norm.mean():.2f}, std: {X_norm.std():.2f}")

    peak_amp = X_norm[:, 0]
    print(f"Пики класса 0 (шум):    среднее {peak_amp[y == 0].mean():.2f}")
    print(f"Пики класса 1 (сигнал): среднее {peak_amp[y == 1].mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # считаем среднее/std НА TRAIN и нормализуем
    X_test_s = scaler.transform(X_test) 
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # === ЭКСПЕРИМЕНТ 1: RF без параллелизма ===
    print("\n=== RF (n_jobs=1, без параллелизма) ===")
    rf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf1.fit(X_train, y_train)
    pred1 = rf1.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred1) * 100:.1f}%")
    print(f"Предсказания: {np.bincount(pred1)}")
    print(f"Важность признаков: {rf1.feature_importances_}")

    # === ЭКСПЕРИМЕНТ 2: RF со всеми признаками ===
    print("\n=== RF (max_features=None) ===")
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42, max_features=None)
    rf2.fit(X_train, y_train)
    pred2 = rf2.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred2) * 100:.1f}%")
    print(f"Предсказания: {np.bincount(pred2)}")

    # === ЭКСПЕРИМЕНТ 3: Logistic Regression (простая линейная модель) ===
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred_lr) * 100:.1f}%")
    print(f"Предсказания: {np.bincount(pred_lr)}")
    print(f"Коэффициенты: {lr.coef_}")

    # === ЭКСПЕРИМЕНТ 4: Gradient Boosting ===
    print("\n=== Gradient Boosting ===")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred_gb) * 100:.1f}%")
    print(f"Предсказания: {np.bincount(pred_gb)}")