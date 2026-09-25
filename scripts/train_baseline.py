import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

from gwlab.dataset import generate_random_sample, load_real_noise
from gwlab.features import extract_features, extract_wavelet_features


def build_dataset(num_samples, real_noise, snr_range):
    """Собирает два набора признаков: базовый (12) и полный (12+16=28)."""
    X_basic, X_full, y = [], [], []
    for _ in range(num_samples):
        noise_src = real_noise if np.random.rand() < 0.5 else None
        _, h, label, _, _ = generate_random_sample(real_noise=noise_src, snr_range=snr_range)
        basic = extract_features(h)
        wav = extract_wavelet_features(h)
        X_basic.append(basic)
        X_full.append(np.concatenate([basic, wav]))
        y.append(label)
    return np.array(X_basic), np.array(X_full), np.array(y)


def run_experiment(name, model, X, y):
    """Честный пайплайн: split → scale (fit на train!) → fit → predict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: Accuracy = {acc * 100:.1f}%  предсказания {np.bincount(pred)}")
    return acc, model, scaler


if __name__ == "__main__":
    print("🧠 Генерируем датасет (SNR 0.5–30.0: сигнал ТИШЕ шума)...")
    real_noise = load_real_noise()
    
    X_basic, X_full, y = build_dataset(8000, real_noise, snr_range=(0.5, 30.0))
    print(f"Форма X_basic: {X_basic.shape}, X_full: {X_full.shape}")
    

    print("\n=== ТОЛЬКО БАЗОВЫЕ ПРИЗНАКИ (12) ===")
    acc_rf, rf_model, rf_scaler = run_experiment(
        "RF  ", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), X_basic, y
    )
    run_experiment("GB  ", GradientBoostingClassifier(n_estimators=200, random_state=42), X_basic, y)
    run_experiment("LR  ", LogisticRegression(random_state=42, max_iter=2000), X_basic, y)

    # Сохраняем ЛУЧШУЮ модель (RF на 12 признаках) для детекции GW150914
    # Папка models/ в КОРНЕ проекта — абсолютный путь, не зависит от cwd
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'rf_baseline.joblib')
    joblib.dump({'model': rf_model, 'scaler': rf_scaler}, model_path)
    print(f"💾 Модель сохранена: {os.path.abspath(model_path)}")

    print("\n=== БАЗОВЫЕ + ВЕЙВЛЕТЫ (28) ===")
    acc_wav, rf_wav, rf_wav_scaler = run_experiment(
        "RF+W", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), X_full, y
    )
    # Сохраняем ЛУЧШУЮ модель для детекции (перезапишет rf_baseline.joblib)
    joblib.dump({'model': rf_wav, 'scaler': rf_wav_scaler}, model_path)
    print(f"💾 Модель RF+W сохранена: {os.path.abspath(model_path)}")

    print("\n=== ТОЛЬКО ВЕЙВЛЕТЫ (16) ===")
    X_wav_only = X_full[:, 12:]
    run_experiment("WAV ", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), X_wav_only, y)

    print("\n=== ТОЛЬКО FFT (4) ===")
    X_fft_only = X_full[:, 6:10]
    run_experiment("FFT ", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1), X_fft_only, y)

    print("\nТоп-10 важных признаков RF+Wavelets:")
    idx = np.argsort(rf_wav.feature_importances_)[::-1][:10]
    for i in idx:
        print(f"   признак №{i}: {rf_wav.feature_importances_[i]:.3f}")