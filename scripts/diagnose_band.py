import os, sys
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
from gwlab.noise import bandpass

h = np.load(os.path.join(project_root, 'data', 'GW150914_H1.npy'))
fs = 4096.0

for lo, hi in [(20, 300), (30, 300), (35, 350), (40, 400)]:
    bp = bandpass(h, fs=fs, f_low=lo, f_high=hi)
    noise_std = bp[:int(8 * fs)].std()                    # до события
    ev = bp[int(14.8 * fs):int(15.2 * fs)]                # слияние (15 с от начала файла)
    peak = np.max(np.abs(ev))
    print(f"полоса {lo}-{hi} Гц: шум std={noise_std:.2e}, пик события={peak:.2e}, "
          f"отношение={peak / noise_std:.1f}")