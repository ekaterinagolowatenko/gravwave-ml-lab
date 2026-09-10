import numpy as np
import matplotlib.pyplot as plt

# Константы
G = 6.67430e-11  # гравитационная постоянная
c = 299792458    # скорость света
M_sun = 1.98847e30  # масса Солнца

# Параметры чёрных дыр
m1, m2 = 30, 30  # 14 сентября 2015 года детекторы LIGO впервые в истории поймали гравитационную волну (событие GW150914). И массы слившихся чёрных дыр были примерно 29 и 36 масс Солнца.
fs = 4096        # частота дискретизации (Гц) Теорема Котельникова (Найквиста): Чтобы корректно оцифровать сигнал, частота дискретизации должна быть минимум в 2 раза выше максимальной частоты в сигнале. Гравитационные волны от таких чёрных дыр заканчиваются на частоте около 200-300 Гц. Значит, нам нужно минимум 600 Гц. 4096 — с огромным запасом.
duration = 2.0   # длительность сигнала (секунды)

print("🌌 Генерация гравитационно-волнового сигнала...")
print(f"Чёрная дыра 1: {m1} M☉")
print(f"Чёрная дыра 2: {m2} M☉")

# Чирп-масса
m1_kg = m1 * M_sun 
m2_kg = m2 * M_sun
mc = ((m1_kg * m2_kg) ** (3/5)) / ((m1_kg + m2_kg) ** (1/5))
print(f"Чирп-масса: {mc / M_sun:.2f} M☉")

# Время
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
tc = duration + 0.005  # время слияния (чуть больше duration)
tau = tc - t  # время до слияния

# Частота сигнала
f = (1 / np.pi) * (5 / (256 * tau)) ** (3/8) * (G * mc / c**3) ** (-5/8)

# Фаза сигнала
phi = 2 * np.pi * np.cumsum(f) / fs

# Амплитуда
amplitude = (tau / tau[0]) ** (-1/4)

# Сигнал
h = amplitude * np.cos(phi)
h = h / np.max(np.abs(h))  # нормировка

print(f"✅ Сигнал сгенерирован!")
print(f"   Длина: {len(t)} точек")
print(f"   Частота в начале: {f[0]:.1f} Гц")
print(f"   Частота в конце: {f[-1]:.1f} Гц")

# График
plt.figure(figsize=(10, 4))
plt.plot(t, h, color='blue', linewidth=0.5)
plt.title('Gravitational Wave Chirp Signal', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (normalized)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chirp_signal.png', dpi=150, bbox_inches='tight')
print(" График сохранён как 'chirp_signal.png'")
plt.show()