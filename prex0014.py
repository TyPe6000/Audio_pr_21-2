## https://jonhyuk0922.tistory.com/114
## https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
## https://librosa.org/doc/latest/index.html
## https://wikidocs.net/92071
## 참고자료 일부 첨부합니당.

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

## wav파일 읽는걸로 찾은거라 wav로만 테스트했습니다. 이외 파일 형식은 몰라요.
filepath = "C:/code/audio_pr/sound/"
filename = "sample0006.wav"

## samplerate 본인 파일에 맞춰주시고
data , samplingrate = librosa.load(filepath + filename, sr=48000) 



## librosa.pyin() Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).
f0 , voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=samplingrate, frame_length=2048)
## librosa.times_like() Return an array of time values to match the time axis from a feature matrix.
times = librosa.times_like(f0, sr=samplingrate)
## 그냥 단순하게 f0,times 축으로 그래프
'''
plt.figure(figsize=(16,9))
plt.plot(times, f0)
plt.grid()
plt.show()
'''
## spectogram 형식. 파란줄로 f0 포현해주는 그래프
'''
fig, ax = plt.subplots()
img = librosa.display.specshow(FTdata, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation') #title
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.show()
'''
#print(f0)
