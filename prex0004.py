import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt

import sounddevice as sd

path = "C:/__file/pcd/audio_pr/sound/"
filename = "sample0001.wav"

samplerate, data = wavfile.read(path + filename)


#time 영역
times = np.arange(len(data))/float(samplerate)
sd.play(data, samplerate)

#print( ' sampling rate : ', samplerate)
#print('time : ', times[-1])

#fft변환 후 주파수 영역?
fft = np.fft.fft(data) / len(data)
fft_abs = abs(fft) # abs : 절대값

# time영역 그래프.
plt.subplot(2,1,1) # subplot()은 여러 개의 그래프를 나타낼때 그 위치를 지정. 
plt.fill_between(times, data)
plt.xlabel("sample")
plt.ylabel("data")
plt.xlim(times[0], times[-1])



# fft후 영역 그래프
plt.subplot(2,1,2)
plt.stem(fft_abs)
plt.show()

