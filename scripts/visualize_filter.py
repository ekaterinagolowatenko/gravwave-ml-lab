import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
b, a = signal.butter(4, 0.01, btype='low')

# 2. Метод freqz вычисляет частотную характеристику фильтра
# w - это частоты, h - это комплексный отклик (амплитуда и фаза)
w, h = signal.freqz(b, a, worN=2000)

# При sample_rate=4096, максимальная частота (Найквиста) = 2048 Гц
sample_rate = 4096
frequencies = w * sample_rate / (2 * np.pi)

# 4. Преобразуем амплитуду в Децибелы (dB), чтобы видеть ослабление
amplitude_db = 20 * np.log10(np.abs(h))

# 5. Рисуем график!
plt.figure(figsize=(10, 5))
plt.plot(frequencies, amplitude_db, color='blue', linewidth=2)
plt.title('Частотная характеристика фильтра Баттерворта (Низкочастотный)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Усиление / Ослабление (дБ)')
plt.grid(True, which='both', linestyle='--')

# Частота среза 0.01 от частоты Найквиста (2048 Гц) = ~20 Гц
cutoff_hz = 0.01 * (sample_rate / 2)
plt.axvline(x=cutoff_hz, color='red', linestyle=':', label=f'Частота среза (~{cutoff_hz:.0f} Гц)')

plt.axhline(y=-3, color='green', linestyle='--', label='-3 дБ (уровень среза)')

plt.legend()
plt.xlim(0, 200) 
plt.savefig('docs/filter_response.png', dpi=150)
