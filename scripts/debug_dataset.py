import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

from gwlab.dataset import generate_random_sample, load_real_noise

noise = load_real_noise()
print(f"Шум: длина={len(noise)}, std={np.std(noise):.3e}, max={np.max(np.abs(noise)):.3e}")

stats = {0: [], 1: []}
for _ in range(200):
    t, h, label, _, _ = generate_random_sample(real_noise=noise)
    stats[label].append(np.max(np.abs(h)))

for lab in [0, 1]:
    arr = np.array(stats[lab])
    print(f"label={lab}: max|h| среднее={arr.mean():.3e}, мин={arr.min():.3e}, макс={arr.max():.3e}")

for target in [1, 0]:
    while True:
        t, h, label, m1, m2 = generate_random_sample(real_noise=noise)
        if label == target:
            break
    plt.figure(figsize=(10, 3))
    plt.plot(t, h, linewidth=0.6)
    plt.title(f"Пример с label={label} (m1={m1:.1f}, m2={m2:.1f})")
    plt.tight_layout()
    plt.savefig(f"docs/debug_example_{label}.png", dpi=120)
    plt.close()

print("💾 Сохранено: docs/debug_example_0.png и docs/debug_example_1.png")