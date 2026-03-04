import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy.io.wavfile import write
import os

# 0. Создание директории для сохранения
output_dir = "files/lab01"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Параметры сигналов
fs = 44100
N_plot = 2048           # Для графиков (быстрый расчет)
duration_audio = 5      # 5 секунд для аудио
N_audio = fs * duration_audio

# Временные оси
t_plot = np.arange(N_plot) / fs
t_audio = np.arange(N_audio) / fs

# Параметры гармоник: x(t) - 110Гц, y(t) - 55Гц
A_x, f0_x, h_x = [1, 0.6, 0.4, 0.2], 110, [1, 2, 3, 4]
A_y, f0_y, h_y = [1, 0.7, 0.5], 55, [1, 2, 3]

def generate_signal(A, f0, h, t):
    s = np.zeros_like(t)
    for Ai, hi in zip(A, h):
        s += Ai * np.sin(2 * np.pi * hi * f0 * t)
    return s

# Генерация
x = generate_signal(A_x, f0_x, h_x, t_plot)
y = generate_signal(A_y, f0_y, h_y, t_plot)

x_long = generate_signal(A_x, f0_x, h_x, t_audio)
y_long = generate_signal(A_y, f0_y, h_y, t_audio)

# 2. Сохранение аудио
def save_wav(filename, data, rate):
    norm_data = data / np.max(np.abs(data))
    path = os.path.join(output_dir, filename)
    write(path, rate, np.int16(norm_data * 32767))
    return path

path_x = save_wav("SigX.wav", x_long, fs)
path_y = save_wav("SigY.wav", y_long, fs)
print(f"Аудио сохранено в: {output_dir}")

# 3. Свои реализации алгоритмов
def my_dft(x):
    N_len = len(x)
    n = np.arange(N_len)
    k = n.reshape((N_len, 1))
    e = np.exp(-2j * np.pi * k * n / N_len)
    return np.dot(e, x)

def my_idft(X):
    N_len = len(X)
    n = np.arange(N_len)
    k = n.reshape((N_len, 1))
    e = np.exp(2j * np.pi * k * n / N_len)
    return np.dot(e, X) / N_len

def my_fft(x):
    N_len = len(x)
    if N_len <= 1: return x
    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N_len) * odd[k] for k in range(N_len // 2)]
    return np.array([even[k] + T[k] for k in range(N_len // 2)] + 
                    [even[k] - T[k] for k in range(N_len // 2)])

def my_ifft(X):
    return np.conj(my_fft(np.conj(X))) / len(X)

def my_convolution(x, y):
    N_len, M_len = len(x), len(y)
    res = np.zeros(N_len + M_len - 1)
    for i in range(len(res)):
        for j in range(max(0, i-M_len+1), min(i+1, N_len)):
            res[i] += x[j] * y[i-j]
    return res

def my_correlation(x, y):
    return my_convolution(x, y[::-1])

# 4. Подготовка данных для графиков
freq = np.arange(N_plot) * fs / N_plot
X_fft_res = my_fft(x)
Y_fft_res = my_fft(y)

# Свертка и корреляция через БПФ (быстрый метод)
N_conv = len(x) + len(y) - 1
N_pow2 = 1 << (N_conv - 1).bit_length()
x_p, y_p = np.pad(x, (0, N_pow2 - N_plot)), np.pad(y, (0, N_pow2 - N_plot))
Z_conv_fft = my_ifft(my_fft(x_p) * my_fft(y_p)).real[:N_conv]
Z_corr_fft_raw = my_ifft(my_fft(x_p) * np.conj(my_fft(y_p))).real
Z_corr_fft = np.fft.fftshift(Z_corr_fft_raw)

# Список всех 24 графиков
plots_data = [
    (0, "1. x(t)", t_plot, x),
    (1, "2. y(t)", t_plot, y),
    (2, "3. x ДПФ Амплитуда", freq[:N_plot//2], 2*np.abs(X_fft_res[:N_plot//2])/N_plot),
    (3, "4. x ДПФ Фаза", freq[:N_plot//2], np.angle(X_fft_res[:N_plot//2])),
    (4, "5. x(t) ОДПФ", t_plot, my_idft(X_fft_res).real),
    (5, "6. x БПФ Амплитуда", freq[:N_plot//2], 2*np.abs(X_fft_res[:N_plot//2])/N_plot),
    (6, "7. x БПФ Фаза", freq[:N_plot//2], np.angle(X_fft_res[:N_plot//2])),
    (7, "8. x(t) ОБПФ", t_plot, my_ifft(X_fft_res).real),
    (8, "9. y ДПФ Амплитуда", freq[:N_plot//2], 2*np.abs(Y_fft_res[:N_plot//2])/N_plot),
    (9, "10. y ДПФ Фаза", freq[:N_plot//2], np.angle(Y_fft_res[:N_plot//2])),
    (10, "11. y(t) ОДПФ", t_plot, my_idft(Y_fft_res).real),
    (11, "12. y БПФ Амплитуда", freq[:N_plot//2], 2*np.abs(Y_fft_res[:N_plot//2])/N_plot),
    (12, "13. y БПФ Фаза", freq[:N_plot//2], np.angle(Y_fft_res[:N_plot//2])),
    (13, "14. y(t) ОБПФ", t_plot, my_ifft(Y_fft_res).real),
    (14, "15. Свертка (Своя)", None, my_convolution(x, y)),
    (15, "16. Свертка через БПФ", None, Z_conv_fft),
    (16, "17. Корреляция (Своя)", None, my_correlation(x, y)),
    (17, "18. Корр. через БПФ", None, Z_corr_fft[(len(Z_corr_fft)-N_conv)//2:][:N_conv]),
    (18, "19. x БПФ Amp (Lib)", freq[:N_plot//2], 2*np.abs(fft(x)[:N_plot//2])/N_plot),
    (19, "20. x БПФ Phase (Lib)", freq[:N_plot//2], np.angle(fft(x)[:N_plot//2])),
    (20, "21. y БПФ Amp (Lib)", freq[:N_plot//2], 2*np.abs(fft(y)[:N_plot//2])/N_plot),
    (21, "22. y БПФ Phase (Lib)", freq[:N_plot//2], np.angle(fft(y)[:N_plot//2])),
    (22, "23. Свертка (Lib)", None, np.convolve(x, y)),
    (23, "24. Корреляция (Lib)", None, signal.correlate(x, y))
]

# 5. Отрисовка в одном окне
fig, axes = plt.subplots(6, 4, figsize=(22, 26)) 
axes = axes.flatten()

for i, title, data_x, data_y in plots_data:
    if data_x is not None:
        axes[i].plot(data_x, data_y, color='tab:blue', linewidth=0.7)
    else:
        axes[i].plot(data_y, color='tab:blue', linewidth=0.7)
    axes[i].set_title(title, fontsize=10, pad=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='both', labelsize=8) 
    axes[i].locator_params(axis='both', nbins=4)  

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)

print("Все расчеты выполнены. Графики построены.")
plt.show()