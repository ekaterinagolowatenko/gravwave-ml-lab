import numpy as np

from gwlab.dataset import generate_random_sample


def test_generate_random_sample_signal_is_distinct_from_noise():
    noise = np.random.normal(0.0, 1e-22, 8192)
    signal_amplitudes = []

    for _ in range(200):
        _, sample, label, _, _ = generate_random_sample(real_noise=noise)
        if label == 1:
            signal_amplitudes.append(float(np.max(np.abs(sample))))

    assert signal_amplitudes, "должен быть хотя бы один пример с сигналом"
    assert float(np.median(signal_amplitudes)) > 5e-20
