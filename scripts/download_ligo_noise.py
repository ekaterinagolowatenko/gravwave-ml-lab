import numpy as np
from gwpy.timeseries import TimeSeries

def get_realistic_noise(duration=4, sample_rate=4096):
    
    try:
        data = TimeSeries.fetch_open_data(
            'H1', 
            '2015-09-12 00:00:00', 
            '2015-09-12 00:00:04', 
            sample_rate=sample_rate
        )
        noise = data.value        
    except Exception as e:
        # Если скачивание не удалось, выполняем этот блок
        print(f"ошибка: {type(e).__name__})")        
        # Генерация цветного шума (упрощенная имитация спектра LIGO)
        # Мы создаем белый шум, а затем фильтруем его, чтобы низкие частоты были громче
        white_noise = np.random.normal(0, 1, int(duration * sample_rate))
        
        # Применяем простой фильтр, чтобы имитировать сейсмический шум на низких частотах
        from scipy import signal
        b, a = signal.butter(4, 0.01, btype='low') # 4 — порядок фильтра (чем выше, тем круче срез) 0.01 — частота среза (нормированная, от 0 до 1) btype='low' — тип фильтра (low = низкие частоты проходят)Коэффициенты фильтра — это числа, которые описывают, как именно фильтр должен обрабатывать сигнал.y[n] = (b[0]*x[n] + b[1]*x[n-1] + ...) / (a[0] + a[1]*y[n-1] + ...)
        noise = signal.filtfilt(b, a, white_noise)
        
        # Нормализуем шум до реальных значений LIGO (порядка 1e-21)
        noise = noise / np.max(np.abs(noise)) * 1e-21
    print(f" Длина шума: {len(noise)} точек")
    print(f" Стандартное отклонение: {np.std(noise):.2e}")
    
    np.save('data/ligo_noise.npy', noise)
    
    return noise

if __name__ == "__main__":
    noise = get_realistic_noise()