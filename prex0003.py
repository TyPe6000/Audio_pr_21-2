import numpy as np
import matplotlib.pyplot as plt
 
fs = 100 
t = np.arange(0, 3, 1 / fs)
f1 = 35 # f1 = 35Hz 
f2 = 10 # f2 = 10Hz 
signal = 0.6 * np.sin(2 * np.pi * f1 * t) + 3 * np.cos(2 * np.pi * f2 * t + np.pi/2)
 
fft = np.fft.fft(signal) / len(signal)  
 
fft_magnitude = abs(fft)