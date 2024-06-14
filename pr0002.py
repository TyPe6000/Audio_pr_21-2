## https://jonhyuk0922.tistory.com/114
## https://iop8890.tistory.com/9
## https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
## https://librosa.org/doc/latest/index.html
## https://wikidocs.net/92071

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd ## 음악 듣기용

## 메인 테스트 대상은 sample0006.wav / 동요 비행기를 120BRM 메트로놈에 맞춰 Ableton Live VST, MIDI Keyboard로 연주한 녹음 파일. 활용 음원 (추정) E_Piano Basic.adg(Ableton Live Lite 기본 포함)
## 입력 시스템 활용 가능성.
filepath = "C:/code/audio_pr/sound/"
filename = "sample0006.wav"

## data : wav file 읽어온 data값. 2-5.Spectogram을 통해 시간, 주파수, 음압에 대한 정보가 있음을 알 수 있다.
## samplingrate : librosa.load(file, sr)에서 sr값은 파이르이 샘플링 주파수다. 이 값을 받아 samplingrate 변수에 저장.
data , samplingrate = librosa.load(filepath + filename, sr=48000) 

## 2. 오디오 파일 이해하기
## 2-1. 기본값 확인

np.set_printoptions(threshold=np.inf)
#print(data)
#print(len(data)) ## len은 array형식인 data의 정보 개수
#print('Sampling rate (Hz): %d' %samplingrate)
#print('Audio length (seconds): %.2f' % (len(data) / samplingrate))


## 2-2.음악 들어보기(생략)

'''
ipd.Audio(y, rate=samplingrate)
'''

## 2-3. 2D 음파 그래프
'''
plt.figure(figsize =(16,6))
librosa.display.waveplot(y=data,sr=samplingrate)
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
librosa.display.specshow(FTDBdata,sr=samplingrate, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()
'''
## 2-6. Mel Spectogram

Meldata = librosa.feature.melspectrogram(data, sr=samplingrate)
Mel_DB = librosa.amplitude_to_db(Meldata, ref=np.max)
'''
plt.figure(figsize=(16,10))
librosa.display.specshow(Mel_DB, sr=samplingrate,hop_length=512, x_axis='time',y_axis='log')
plt.colorbar()
plt.show()
'''
## 3. 오디오 특성 추출(Audio Feature Extraction)
## 3-1. Tempo(BPM)

ex_tempo , _ = librosa.beat.beat_track(data,sr=samplingrate)
## 읽기 쉽도록 반올림
BPM = round(ex_tempo)

## Unit_Beat는 
if not BPM == 0 :
    Unit_Beat = int((60/BPM)*samplingrate/4)
'''
print(ex_tempo)
print(BPM) ## 정수화된 BPM
print(Unit_Beat) ## 16분음표 Beat_Time
'''

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
## librosa.pyin() Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).
## 얘는 그냥 librosa documentation에서 찾아서 썼음.
## y=입력값(ndarray형식 배열 데이터), 
#f0 , voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C#8'), sr=samplingrate, frame_length=2048)
f0 = librosa.yin(data, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C8"), sr=samplingrate, frame_length=int(Unit_Beat/16), win_length=None, hop_length=None)
'''
f0, voiced_flag, voiced_probs = librosa.pyin(y=data, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C8"), sr=samplingrate, \
                                                frame_length=Unit_Beat/16, win_length=None, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), \
                                                boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, \
                                                fill_na=np.nan, center=True, pad_mode='reflect')
'''
## librosa.times_like() Return an array of time values to match the time axis from a feature matrix.
times = librosa.times_like(f0, sr=samplingrate)
'''
plt.figure(figsize=(16,9))
plt.plot(times, f0)
plt.grid()
plt.show()
'''
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

## f0에 대한 log연산으로 구간간격 1인 형태로 설정 후, numpy의 내림연산 함수 trunc를 활용해 양자화
## 정수값 형태로 출력 가능. A0의 음계 번호를 0으로 설정해서 피아노 연주 가능한 최소음인 A0이상의 모든음이 0 또는 양수 가 되도록 했다. 

Qt_f0 = np.floor(12*np.log2(f0/440)+0.5) + 48 ## 내림연산 활용. (한옥타브 높게 나오는 오류는 여기서 내림이 아니라 버림연산을 해서인 것으로 확인)

#print(Qt_f0)

plt.plot(times, Qt_f0)
plt.grid(True,'both','both') 
plt.show()

