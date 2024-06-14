import numpy as np
import scipy.signal
import scipy.stats
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd ## 음악 듣기용

np.set_printoptions(threshold=np.inf)

### 01. 음악 파일 입력

## 메인 테스트 대상은 sample0006.wav / 동요 비행기를 120BRM 메트로놈에 맞춰 Ableton Live VST, MIDI Keyboard로 연주한 녹음 파일. 활용 VST  E_Piano Basic.adg(Ableton Live Lite 기본 포함)
## 입력 시스템 활용 가능성.
filepath = "C:/code/audio_pr/sound/"
filename = "sample0012.wav"

## data : wav file 읽어온 data값. 2-5.Spectogram을 통해 시간, 주파수, 음압에 대한 정보가 있음을 알 수 있다.
## samplingrate : librosa.load(file, sr)에서 sr값은 파일의 샘플링 주파수다. 이 값을 받아 samplingrate 변수에 저장.
data , samplingrate = librosa.load(filepath + filename, sr=48000) 


### 02. BPM 검출

ex_tempo , _ = librosa.beat.beat_track(data,sr=samplingrate)
## 읽기 쉽도록 반올림
BPM = round(ex_tempo)

## Unit_Beat는 
if not BPM == 0 :
    Unit_Beat = int((60/BPM)*samplingrate/4)

#print(ex_tempo)
#print(BPM) ## 정수화된 BPM
#print(Unit_Beat) ## 16분음표 Beat_Time에 해당하는 sample 수


### 03. 기본 주파수 검출

##
## Unit_Beat/16의 frame_length로 추출했기 때문에 f0값 파생에서는 Unit_Beat만큼의 데이터가 16개가 된다.
f0 = librosa.yin(data, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C#8"), sr=samplingrate, frame_length=int(Unit_Beat/16), win_length=None, hop_length=None)


## librosa.times_like() Return an array of time values to match the time axis from a feature matrix.
## f0에 대응하는 시간 단위
times = librosa.times_like(f0, sr=samplingrate)

## f0에 대한 log연산으로 구간간격 1인 형태로 설정 후, numpy의 내림연산 함수 trunc를 활용해 양자화
## 정수값 형태로 출력 가능. A0의 음계 번호를 0으로 설정해서 피아노 연주 가능한 최소음인 A0이상의 모든음이 0 또는 양수 가 되도록 했다. 

Q_f0 = np.floor(12*np.log2(f0/440)+0.5) + 48

#print(Q_f0)

#plt.plot(times, Q_f0)
#plt.grid(True)
#plt.show()

### 04. pitch 설정

## 최빈값 검출 알고리즘인 scipy.stats.mode()를 통해 Unit_Beat 단위로 최빈값 검출 후 note_pitch에 저장.
t = 0
moded_pitch = np.array([])

for  t in range(len(f0)) :
    
    pitch = (scipy.stats.mode(Q_f0[t:t+16])[0]) 
    moded_pitch = np.concatenate((moded_pitch, pitch), axis=0)
    t = t + 16

print(moded_pitch)
print(len(moded_pitch))

plt.plot(times, moded_pitch)
plt.grid(True)
plt.show()


### 05. note 분할 + note 정보 정리.

## ★★★★★  여기 수정 필요!!!!!!!!!!!!!!!! 오류 해결 안됨.

## Q_f0에서 peak 검출. 전 Unit_Beat 구간과 음이 같고 peak가 없으면 연결해서 같은 note로 취급. peak 발견시 그 시점에서 note 연결 종료 후 다음 note로 넘어간다.
note_pitch = np.array([])
note_position = np.array([])
note_size = np.array([])
note_info = np.array([])
empty = np.array([])
position = 0
size = 1

for  position in range(int(len(f0)/16)) :
    
    ## 현재 칸의 뒷부분에서 peak가 발생하는 경우, 현재 칸까지를 저장하고 다음 note정보 생성
    #late_peaks, l_properties = scipy.signal.find_peaks(Q_f0[position:position+16*size], height=1)
    ## 다음 칸의 앞 절반 부분 내에서 peak가 발생하는 경우, 현재 칸까지를 저장하고 다음 note정보 생성
    #early_peaks, e_properties = scipy.signal.find_peaks(Q_f0[position:position+16*size+8], height=1)
    Q_f0[position:position+16*size]

    

    if late_peaks == empty :

            size = size + 1

    elif early_peaks == empty :

            size = size + 1

    else :
            note_pitch = np.concatenate((note_pitch, [moded_pitch[position]]), axis=0)
            note_position = np.concatenate((note_position, [position]), axis=0)
            note_size = np.concatenate((note_size, [size]), axis=0)
            #note_info =  np.append(note_info, np.array([int(moded_pitch[position]), position, size]))

            position = position + size

            size = 1


#print(note_pitch)
#print(note_position)
#print(note_size)
#print(note_info)


### 06. 구축된 data를 통해 악보 작성 시각화하는 알고리즘