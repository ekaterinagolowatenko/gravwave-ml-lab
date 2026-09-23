import os
import time
import requests
import numpy as np
from gwpy.timeseries import TimeSeries

print("📥 Попытка скачать GW150914 с retry-логикой...")

# Параметры события GW150914
start_time = "2015-09-14 09:50:30"  # берём чуть больше окна
end_time = "2015-09-14 09:50:50"
ifo = "H1"

max_attempts = 3
for attempt in range(1, max_attempts + 1):
    print(f"\nПопытка {attempt}/{max_attempts}...")
    try:
        # Увеличиваем timeout до 5 минут
        data = TimeSeries.fetch_open_data(
            ifo, start_time, end_time, 
            cache=True,
            timeout=300  # 5 минут timeout
        )
        print(f"✅ Успех! Скачано {len(data)} отсчётов, частота {data.sample_rate} Гц")
        
        # Сохраняем в numpy для быстрого чтения
        os.makedirs('data', exist_ok=True)
        np.save('data/GW150914_H1.npy', data.value)
        np.save('data/GW150914_H1_meta.npy', {
            'sample_rate': float(data.sample_rate.value),
            'start_time': data.t0.value
        })
        print("💾 Сохранено: data/GW150914_H1.npy")
        break
        
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {str(e)[:100]}")
        