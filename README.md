# GravWave ML Lab

Machine learning for gravitational wave detection and analysis

Интерактивная лаборатория для моделирования и анализа гравитационно-волновых сигналов от сливающихся чёрных дыр.

## Цель проекта

Создание ML-системы для обнаружения и оценки параметров гравитационно-волновых сигналов от двойных чёрных дыр.

## Что внутри

- **Генерация сигналов**: моделирование гравитационных волн (событие типа GW150914)
- **Визуализация**: графики волны, эволюции частоты и спектрограммы
- **Физика**: расчёт chirp mass, фазы и амплитуды

## Быстрый старт

```bash
# Клонируем репозиторий
git clone https://github.com/ekaterinagolowatenko/gravwave-ml-lab.git
cd gravwave-ml-lab

# Устанавливаем зависимости
pip install numpy scipy matplotlib jupyter

# Запускаем ноутбук
jupyter notebook 01_chirp_signal.ipynb
