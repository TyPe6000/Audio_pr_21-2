## https://jonhyuk0922.tistory.com/114
## https://iop8890.tistory.com/9
## https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
## https://librosa.org/doc/latest/index.html
## https://wikidocs.net/92071

import numpy as np
import scipy.signal
import scipy.stats
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd ## 음악 듣기용

## 메인 테스트 대상은 sample0006.wav / 동요 비행기를 120BRM 메트로놈에 맞춰 Ableton Live VST, MIDI Keyboard로 연주한 녹음 파일. 활용 VST  E_Piano Basic.adg(Ableton Live Lite 기본 포함)
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

ex_tempo , _ = librosa.beat.beat_track(data,sr=samplingrate)
## 읽기 쉽도록 반올림
BPM = round(ex_tempo)

## Unit_Beat는 
if not BPM == 0 :
    Unit_Beat = int((60/BPM)*samplingrate/4)

#print(ex_tempo)
#print(BPM) ## 정수화된 BPM
#print(Unit_Beat) ## 16분음표 Beat_Time에 해당하는 sample 수

## librosa.pyin() Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).
## 얘는 그냥 librosa documentation에서 찾아서 썼음.
## y=입력값(ndarray형식 배열 데이터), 
#f0 , voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=samplingrate, frame_length=2048)
## Unit_Beat/16의 frame_length로 추출했기 때문에 f0값 파생에서는 Unit_Beat만큼의 데이터가 16개가 된다.
f0 = librosa.yin(data, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C#8"), sr=samplingrate, frame_length=int(Unit_Beat/16), win_length=None, hop_length=None)


## librosa.times_like() Return an array of time values to match the time axis from a feature matrix.
times = librosa.times_like(f0, sr=samplingrate)

            #print(len(f0)) ## len(f0) = len(data)/(int(Unit_Beat/16))
            #print(len(times))
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

Q_f0 = np.floor(12*np.log2(f0/440)+0.5) + 48
'''
print(Q_f0)

plt.plot(times, Q_f0)
plt.grid(True)
plt.show()
'''

            #peaks, properties = scipy.signal.find_peaks(Q_f0, height=0)

            #print(f'Index of each peaks : {peaks}')
            #print(f'Height of each peaks : {properties["peak_heights"]}')

#peak_point = int(np.around(peaks/6000))


## 최빈값 검출 알고리즘인 scipy.stats.mode()를 통해 Unit_Beat 단위로 최빈값 검출 후 note_pitch에 저장.
t = 0
moded_pitch = np.array([])

for  t in range(len(f0)) :
    
    pitch = (scipy.stats.mode(Q_f0[t:t+16])[0]) 
    moded_pitch = np.concatenate((moded_pitch, pitch), axis=0)
    t = t + 16

            #print(moded_pitch)
            #print(len(moded_pitch))
'''
plt.plot(times, moded_pitch)
plt.grid(True)
plt.show()
'''




## Q_f0에서 peak 검출. 전 Unit_Beat 구간과 음이 같고 peak가 없으면 연결해서 같은 note로 취급. peak 발견시 그 시점에서 note 연결 종료 후 다음 note로 넘어간다.
note_pitch = np.array([])
note_position = np.array([])
note_size = np.array([])
empty = np.array([])
position = 0
size = 1

for  position in range(len(f0)) :
    
    ## 현재 칸의 뒷부분에서 peak가 발생하는 경우, 현재 칸까지를 저장하고 다음 note정보 생성
    late_peaks, properties = scipy.signal.find_peaks(Q_f0[position:position+size], height=0)
    ## 다음 칸의 앞 절반 부분 내에서 peak가 발생하는 경우, 현재 칸까지를 저장하고 다음 note정보 생성
    early_peaks, properties = scipy.signal.find_peaks(Q_f0[position:position+size+8], height=0)
    
    if late_peaks == empty and early_peaks == empty :

            size = size + 1
            position = position

    else :
            note_pitch = np.concatenate((note_pitch, [moded_pitch[position]]), axis=0)
            note_position = np.concatenate((note_position, [position]), axis=0)
            note_size = np.concatenate((note_size, [size]), axis=0)

            position = position + size

            size = 1


print(note_pitch)
print(note_position)
print(note_size)


