import numpy as np
from .constants import G, C, M_SUN
from .physics import get_chirp_mass

def generate_chirp(m1, m2, fs=4096, duration=2.0):
    m1_kg = m1 * M_SUN
    m2_kg = m2 * M_SUN
    chirp_mass = get_chirp_mass(m1, m2)
    
    t = np.linspace(0, duration, int(fs * duration))
    tc = duration + 0.05     # tc — момент слияния (чуть позже конца сигнала, чтобы избежать деления на 0)
    tau = tc - t    # tau — время, оставшееся до слияния (обратный отсчёт)
    
    f_t = (1 / np.pi) * (5 / 256 / tau) ** (3/8) * (G * chirp_mass / C**3) ** (-5/8)
    phi_t = 2 * np.pi * np.cumsum(f_t) / fs    # phi_t — фаза колебаний (накапливается из частоты)
    amplitude = tau ** (-1/4)
    h_t = amplitude * np.cos(phi_t)    # h_t — итоговый сигнал гравитационной волны
    
    return t, h_t