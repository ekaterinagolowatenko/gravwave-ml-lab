import sys
import numpy as np

# Добавляем путь к модулю, так как скрипт лежит в папке scripts, а не в корне
sys.path.insert(0, '../src')

from gwlab.dataset import generate_random_sample

def create_dataset(num_samples=1000, fs=4096, duration=2.0):
    print(f"Начинаем генерацию датасета из {num_samples} примеров...")
    
    # Создаем пустые списки для хранения данных
    all_signals = []
    all_labels = []
    all_m1 = []
    all_m2 = []
    
    for i in range(num_samples):
        # Вызываем нашу функцию генерации
        t, noisy_h, label, true_m1, true_m2 = generate_random_sample(fs, duration)
        
        # Сохраняем результаты в списки
        all_signals.append(noisy_h)
        all_labels.append(label)
        all_m1.append(true_m1)
        all_m2.append(true_m2)
        
        # Печатаем прогресс каждые 100 примеров
        if (i + 1) % 100 == 0:
            print(f"Сгенерировано {i + 1} примеров...")
            
    # Превращаем списки в массивы NumPy (так быстрее и удобнее для ML)
    X = np.array(all_signals)
    y_labels = np.array(all_labels)
    y_m1 = np.array(all_m1)
    y_m2 = np.array(all_m2)
    
    # Сохраняем в файлы
    np.save('../data/X_signals.npy', X)
    np.save('../data/y_labels.npy', y_labels)
    np.save('../data/y_m1.npy', y_m1)
    np.save('../data/y_m2.npy', y_m2)
    
    print("✅ Датасет успешно сохранен в папку data/")
    print(f"Форма массива сигналов X: {X.shape}")

if __name__ == "__main__":
    create_dataset(num_samples=1000)
    