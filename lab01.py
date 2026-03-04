import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy.io.wavfile import write
import os

output_dir = "files/lab01"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Параметры сигналов
fs = 44100
N_plot = 2048       
duration_audio = 5  
N_audio = fs * duration_audio

# Временные оси для графиков и аудио
t_plot = np.arange(N_plot) / fs
t_audio = np.arange(N_audio) / fs

# Параметры гармоник
A_x, f0_x, h_x = [1, 0.6, 0.4, 0.2], 110, [1, 2, 3, 4]
A_y, f0_y, h_y = [1, 0.7, 0.5], 55, [1, 2, 3]

# Генерация сигналов
def generate_signal(A, f0, h, t):
    s = np.zeros_like(t)
    for Ai, hi in zip(A, h):
        s += Ai * np.sin(2 * np.pi * hi * f0 * t)
    return s

x = generate_signal(A_x, f0_x, h_x, t_plot)
y = generate_signal(A_y, f0_y, h_y, t_plot)

x_long = generate_signal(A_x, f0_x, h_x, t_audio)
y_long = generate_signal(A_y, f0_y, h_y, t_audio)

# Сохранение аудио
def save_wav(filename, data, rate):
    norm_data = data / np.max(np.abs(data))
    path = os.path.join(output_dir, filename)
    write(path, rate, np.int16(norm_data * 32767))
    return path

path_x = save_wav("SigX.wav", x_long, fs)
path_y = save_wav("SigY.wav", y_long, fs)

# Реализации алгоритмов
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

X_dft_res = my_dft(x)
Y_dft_res = my_dft(y)

X_fft_res = my_fft(x)
Y_fft_res = my_fft(y)

# Свертка и корреляция через БПФ
N_conv = len(x) + len(y) - 1
N_pow2 = 1 << (N_conv - 1).bit_length()
x_p, y_p = np.pad(x, (0, N_pow2 - N_plot)), np.pad(y, (0, N_pow2 - N_plot))
Z_conv_fft = my_ifft(my_fft(x_p) * my_fft(y_p)).real[:N_conv]
Z_corr_fft_raw = my_ifft(my_fft(x_p) * np.conj(my_fft(y_p))).real
Z_corr_fft = np.fft.fftshift(Z_corr_fft_raw)

# Список графиков
plots_data = [
    (0, "1. x(t)", t_plot, x, None),
    (1, "5. x(t) ОДПФ", t_plot, my_idft(X_dft_res).real, None),
    (2, "8. x(t) ОБПФ", t_plot, my_ifft(X_fft_res).real, None),
    (3, "15. Свертка", None, my_convolution(x, y), None),
    
    (4, "19. x(t) БПФ Амплитудный спектр (Lib)", freq[:N_plot//2], 2*np.abs(fft(x)[:N_plot//2])/N_plot, 1000),
    (5, "3. x(t) ДПФ Амплитудный спектр", freq[:N_plot//2], 2*np.abs(X_dft_res[:N_plot//2])/N_plot, 1000),
    (6, "6. x(t) БПФ Амплитудный спектр", freq[:N_plot//2], 2*np.abs(X_fft_res[:N_plot//2])/N_plot, 1000),
    (7, "16. Свертка через БПФ", None, Z_conv_fft, None),
    
    (8, "20. x(t) БПФ Фазовый спектр (Lib)", freq[:N_plot//2], np.angle(fft(x)[:N_plot//2]), 1000),
    (9, "4. x(t) ДПФ Фазовый спектр", freq[:N_plot//2], np.angle(X_dft_res[:N_plot//2]), 1000),
    (10, "7. x(t) БПФ Фазовый спектр", freq[:N_plot//2], np.angle(X_fft_res[:N_plot//2]), 1000),
    (11, "23. Свертка (Lib)", None, np.convolve(x, y), None),
    
    (12, "2. y(t)", t_plot, y, None),
    (13, "11. y(t) ОДПФ", t_plot, my_idft(Y_dft_res).real, None),
    (14, "14. y(t) ОБПФ", t_plot, my_ifft(Y_fft_res).real, None),
    (15, "17. Корреляция", None, my_correlation(x, y), None),
    
    (16, "21. y(t) БПФ Амплитудный спектр (Lib)", freq[:N_plot//2], 2*np.abs(fft(y)[:N_plot//2])/N_plot, 500),
    (17, "9. y(t) ДПФ Амплитудный спектр", freq[:N_plot//2], 2*np.abs(Y_dft_res[:N_plot//2])/N_plot, 500),
    (18, "12. y(t) БПФ Амплитудный спектр", freq[:N_plot//2], 2*np.abs(Y_fft_res[:N_plot//2])/N_plot, 500),
    (19, "18. Корреляция через БПФ", None, Z_corr_fft[(len(Z_corr_fft)-N_conv)//2:][:N_conv], None),
    
    (20, "22. y(t) БПФ Фазовый спектр (Lib)", freq[:N_plot//2], np.angle(fft(y)[:N_plot//2]), 500),
    (21, "10. y(t) ДПФ Фазовый спектр", freq[:N_plot//2], np.angle(Y_dft_res[:N_plot//2]), 500),
    (22, "13. y(t) БПФ Фазовый спектр", freq[:N_plot//2], np.angle(Y_fft_res[:N_plot//2]), 500),
    (23, "24. Корреляция (Lib)", None, signal.correlate(x, y), None)
]

# Отрисовка 
fig, axes = plt.subplots(6, 4, figsize=(22, 26)) 
axes = axes.flatten()

for i, title, data_x, data_y, x_lim in plots_data:
    
    if "x(t)" in title:
        line_color = 'tab:blue'
    elif "y(t)" in title:
        line_color = 'tab:red'
    else:
        line_color = 'tab:purple'
    
    if data_x is not None:
        axes[i].plot(data_x, data_y, color=line_color, linewidth=0.7)
    else:
        axes[i].plot(data_y, color=line_color, linewidth=0.7)
    
    if x_lim is not None:
        axes[i].set_xlim(0, x_lim)
    
    axes[i].set_title(title, fontsize=10, pad=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='both', labelsize=8) 
    axes[i].locator_params(axis='both', nbins=4)  

plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04, hspace=0.6, wspace=0.2)
plt.show()