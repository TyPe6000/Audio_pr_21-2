## https://jonhyuk0922.tistory.com/114
## https://iop8890.tistory.com/9
## https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
## https://librosa.org/doc/latest/generated/librosa.display.specshow.html#librosa.display.specshow

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

## 메인 테스트 대상은 sample0006.wav / 동요 비행기를 120BRM 메트로놈에 맞춰 Ableton Live VST, MIDI Keyboard로 연주한 녹음 파일. 활용 음원 (추정) E_Piano Basic.adg(Ableton Live Lite 기본 포함)
filepath = "C:/code/audio_pr/sound/"
filename = "C3Sound.wav"

data , sr = librosa.load(filepath + filename, sr=48000) 

## 2. 오디오 파일 이해하기
## 2-1. 기본값 확인
'''
print(data)
print(len(data))
print('Sampling rate (Hz): %d' %sr)
print('Audio length (seconds): %.2f' % (len(data) / sr))
'''

## 2-2.음악 들어보기(생략)
'''
import IPython.display as ipd
ipd.Audio(y, rate=sr)
'''

## 2-3. 2D 음파 그래프
'''
plt.figure(figsize =(16,6))
librosa.display.waveplot(data=data,sr=sr)
plt.show()
'''

## 2-4. Fourier Transform(푸리에 변환)

FTdata = np.abs(librosa.stft(data, n_fft=2048, hop_length=512)) #n_fft : window size / 이 때, 음성의 길이를 얼마만큼으로 자를 것인가? 를 window라고 부른다.

'''
print(FTdata.shape)

plt.figure(figsize=(16,6))
plt.plot(FTdata)
plt.show()
'''
## 2-5.Spectogram

FTDBdata = librosa.amplitude_to_db(FTdata, ref=np.max) #amplitude(진폭) -> DB(데시벨)로 바꿔라
'''
plt.figure(figsize=(16,6))
librosa.display.specshow(FTDBdata,sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()
'''
## 2-6. Mel Spectogram

Meldata = librosa.feature.melspectrogram(data, sr=sr)
Mel_DB = librosa.amplitude_to_db(Meldata, ref=np.max)
'''
plt.figure(figsize=(16,10))
librosa.display.specshow(Mel_DB, sr=sr,hop_length=512, x_axis='time',y_axis='log')
plt.colorbar()
plt.show()
'''
## 3. 오디오 특성 추출(Audio Feature Extraction)
## 3-1. Tempo(BPM)

ex_tempo , _ = librosa.beat.beat_track(data,sr=sr)
## 읽기 쉽도록 반올림
BPM = round(ex_tempo)
print(BPM)
## Unit_Beat는 
Unit_Beat = (60/BPM)/4

## 3-2. Zero Crossing Rate
## * 음파가 양에서 음으로 또는 음에서 양으로 바뀌는 비율
## * 간단하지만 많이 쓰인다.
'''
zero_crossings = librosa.zero_crossings(y, pad=False)

print(zero_crossings)
print(sum(zero_crossings)) # 음 <-> 양 이동한 횟수
'''

## test : data.dim = 1 (1차원 배열) data.shape = 1027776
'''
print(data)
print(data.ndim)
print(data.shape)
'''
## test : Ftdata.dim = 2 (2차원 배열)
'''
print(Ftdata)
print(Ftdata.ndim)
print(Ftdata.shape)
'''
## test : DB.dim = 2 (2차원 배열)

'''
print(FTDBdata)
print(FTDBdata.ndim)
print(FTDBdata.shape)
'''

f0 , voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=48000, frame_length=2048)
times = librosa.times_like(f0)


fig, ax = plt.subplots()
img = librosa.display.specshow(FTdata, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation') #title
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.show()

#print(f0)

