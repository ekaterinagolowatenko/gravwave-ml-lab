# GravWave ML Lab
**Machine learning for gravitational wave detection and analysis**

Интерактивная лаборатория для моделирования и анализа гравитационно-волновых сигналов от сливающихся чёрных дыр, с ML-детектором, обученным на синтетике и проверенным на реальном событии GW150914.

## Цель проекта

Создание ML-системы для обнаружения гравитационно-волновых сигналов от двойных чёрных дыр в записях детекторов LIGO. Система обучается на синтетических чирпах с добавлением реального шума и тестируется на историческом событии GW150914 (первая зарегистрированная гравитационная волна, 14 сентября 2015 г.).

## Что внутри

**Физика и генерация сигналов:**
- Генератор чирпа от двойной чёрной дыры (`src/gwlab/waveforms.py`)
- Расчёт chirp mass, фазовой эволюции и амплитудной огибающей
- Визуализация: волновая форма, эволюция частоты, спектрограмма
- Стартовый ноутбук `01_chirp_signal.ipynb`

**ML-пайплайн детекции:**
- Генератор датасета: чирп + реальный шум LIGO с логарифмической шкалой SNR 0.5–30 (`src/gwlab/dataset.py`)
- 12 базовых + 16 вейвлет-признаков (`src/gwlab/features.py`)
- Сравнение моделей: RF, Gradient Boosting, Logistic Regression, абляция по признакам (`scripts/train_baseline.py`)
- Детекция на реальных данных скользящим окном (`scripts/detect_gw150914.py`)
- Диагностики: распределение SNR по полосам, сравнение признаков train/serve (`scripts/diagnose_*.py`)

## Структура проекта

```text
gravwave-ml-lab/
├── 01_chirp_signal.ipynb     — знакомство с физикой чирпа
├── src/gwlab/
│   ├── waveforms.py          — генератор чирпа
│   ├── features.py           — извлечение признаков (FFT + вейвлеты)
│   ├── dataset.py            — генератор обучающих примеров
│   └── noise.py              — единая функция band-pass для train и serve
├── scripts/
│   ├── train_baseline.py     — обучение и сравнение моделей
│   ├── detect_gw150914.py    — детекция на реальной записи
│   └── diagnose_*.py         — скрипты диагностики
├── data/                     — реальный шум LIGO и запись GW150914 (не в git, .gitignore)
├── models/                   — сохранённые модели (не в git)
└── docs/                     — графики детекции и спектров
```

Быстрый запуск:

git clone https://github.com/ekaterinagolowatenko/gravwave-ml-lab.git
cd gravwave-ml-lab

# Зависимости
pip install numpy scipy matplotlib gwpy scikit-learn joblib pywt

# 1. Знакомство с физикой чирпа
jupyter notebook 01_chirp_signal.ipynb

# 2. Скачать реальные данные LIGO (GW150914)
python3 scripts/download_gw150914_v2.py

# 3. Обучить модель (2–3 минуты)
python3 scripts/train_baseline.py

# 4. Запустить детекцию на реальном событии
python3 scripts/detect_gw150914.py
# → график в docs/gw150914_detection.png, пик вероятности в момент слияния

## Результаты
После финальной починки пайплайна модель RandomForest с базовыми + вейвлет-признаками (accuracy ≈ 84% на синтетике с честной шкалой SNR 0.5–30) выдаёт пик вероятности детекции на моменте слияния чёрных дыр (t = 5.00 с) в записи GW150914:
<img width="859" height="345" alt="image" src="https://github.com/user-attachments/assets/6c3efc8e-1018-46c2-8969-dbfedfce4455" />
