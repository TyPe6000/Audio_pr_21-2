import numpy as np
import scipy.signal
import scipy.stats
import librosa
import matplotlib.pyplot as plt
import librosa.display
import cv2

np.set_printoptions(threshold=np.inf)

### 01. 음악 파일 입력

## 메인 테스트 대상은 sample0006.wav / 동요 비행기를 120BRM 메트로놈에 맞춰 Ableton Live VST, MIDI Keyboard로 연주한 녹음 파일.
## 활용 VST  E_Piano Basic.adg(Ableton Live Lite 기본 포함)
## 입력 시스템 활용 가능성.
filepath = "C:/code/audio_pr/sound/"
filename = "B120.wav"

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
print(BPM) ## 정수화된 BPM
print(Unit_Beat) ## 16분음표 Beat_Time에 해당하는 sample 수


### 03. 기본 주파수 검출


## Unit_Beat/8의 frame_length로 추출했기 때문에 f0값 파생에서는 Unit_Beat만큼의 구간내의 데이터가 8개가 된다.
f0 = librosa.yin(data, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C#8"), sr=samplingrate, frame_length=int(Unit_Beat/16), win_length=None, hop_length=None)


## librosa.times_like() Return an array of time values to match the time axis from a feature matrix.
## f0에 대응하는 시간 단위
times = librosa.times_like(f0, sr=samplingrate)

## f0에 대한 log연산으로 구간간격 1인 형태로 설정 후, numpy의 내림연산 함수 floor를 활용해 양자화
## 정수값 형태로 출력 가능. A0의 음계 번호를 0으로 설정해서 피아노 연주 가능한 최소음인 A0이상의 모든음이 0 또는 양수 가 되도록 했다. 

Q_f0 = np.floor(12*np.log2(f0/440)+0.5) + 48

#print(Q_f0)

#plt.plot(times, Q_f0)
#plt.grid(True)
#plt.show()
#print(len(f0))
#print(len(f0)/16)
#print(int(len(f0)/16))
### 04. pitch 설정

## 최빈값 검출 알고리즘인 scipy.stats.mode()를 통해 Unit_Beat 단위로 최빈값 검출 후 note_pitch에 저장.
t = 0
moded_pitch = np.array([])

for t in range(int(len(Q_f0)/16)) :
    t = t + 1    
    if t < int(len(Q_f0)/16) :
        mode = (scipy.stats.mode(Q_f0[16*t:16*(t+1)])[0])
        moded_pitch = np.hstack([moded_pitch, mode])
        
#print(mode)



#print(moded_pitch)
#print(len(moded_pitch))

#plt.plot(moded_pitch)
#plt.grid(True)
#plt.show()


### 05. note 분할 + note 정보 정리.

## Q_f0에서 peak 검출. 전 Unit_Beat 구간과 음이 같고 peak가 없으면 연결해서 같은 note로 취급. peak 발견시 그 시점에서 note 연결 종료 후 다음 note로 넘어간다.

note_cut = np.array([])

peaks, properties = scipy.signal.find_peaks(Q_f0, height=0)
print(peaks)

#u = 0

#for u in range(len(peaks)) :
        #u = u + 1
        
note_cut = np.around(peaks/16)
note_cut = note_cut.astype(int)
print(note_cut)


### 06. 구축된 data를 통해 악보 작성 시각화하는 알고리즘


imgpath = "C:/code/audio_pr/score/"
imgfile = "emptyscore.png"

## 공백 오선보 샘플 load

score = cv2.imread(imgpath + imgfile, cv2.IMREAD_GRAYSCALE)

## note 종류별로 image load해서 좌표계에 띄울 수 있도록 준비

def setwholenote(hpos, vpos) :

        Xsize = 30
        Ysize = 22

        whole = cv2.imread(imgpath + "wholenote.png", cv2.IMREAD_GRAYSCALE)
        note_whole = cv2.resize(whole, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_whole)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def sethalfnote(hpos, vpos) :

        Xsize = 30
        Ysize = 94

        half = cv2.imread(imgpath + "halfnote.png", cv2.IMREAD_GRAYSCALE)
        note_half = cv2.resize(half, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_half)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def sethalfnoterev(hpos, vpos) :

        Xsize = 30
        Ysize = 94

        halfrev = cv2.imread(imgpath + "halfnoterev.png", cv2.IMREAD_GRAYSCALE)
        note_halfrev = cv2.resize(halfrev, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_halfrev)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def setquarternote(hpos, vpos) :

        Xsize = 30
        Ysize = 94

        quarter = cv2.imread(imgpath + "quarternote.png", cv2.IMREAD_GRAYSCALE)
        note_quarter = cv2.resize(quarter, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_quarter)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def setquarternoterev(hpos, vpos) :

        Xsize = 30
        Ysize = 94

        quarterrev = cv2.imread(imgpath + "quarternoterev.png", cv2.IMREAD_GRAYSCALE)
        note_quarterrev = cv2.resize(quarterrev, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_quarterrev)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set8thnote(hpos, vpos) :

        Xsize = 50
        Ysize = 94

        eighth = cv2.imread(imgpath + "8thnote.png", cv2.IMREAD_GRAYSCALE)
        note_8th = cv2.resize(eighth, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_8th)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set8thnoterev(hpos, vpos) :

        Xsize = 50
        Ysize = 94

        eighthrev = cv2.imread(imgpath + "8thnoterev.png", cv2.IMREAD_GRAYSCALE)
        note_8threv = cv2.resize(eighthrev, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_8threv)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set16thnote(hpos, vpos) :

        Xsize = 50
        Ysize = 94

        sixteenth = cv2.imread(imgpath + "16thnote.png", cv2.IMREAD_GRAYSCALE)
        note_16th = cv2.resize(sixteenth, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_16th)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set16thnoterev(hpos, vpos) :

        Xsize = 50
        Ysize = 94

        sixteenthrev = cv2.imread(imgpath + "8thnoterev.png", cv2.IMREAD_GRAYSCALE)
        note_sixteenthrev = cv2.resize(sixteenthrev, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, note_sixteenthrev)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def addnotedot(hpos, vpos) :
        
        Xsize = 4
        Ysize = 4

        d = cv2.imread(imgpath + "dot.png", cv2.IMREAD_GRAYSCALE)
        dot = cv2.resize(d, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, dot)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def setwholerest(hpos, vpos) :
        
        Xsize = 30
        Ysize = 22

        rwhole = cv2.imread(imgpath + "wholerest.png", cv2.IMREAD_GRAYSCALE)
        rest_whole = cv2.resize(rwhole, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, rest_whole)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def sethalfrest(hpos, vpos) :
        
        Xsize = 30
        Ysize = 24

        rhalf = cv2.imread(imgpath + "halfrest.png", cv2.IMREAD_GRAYSCALE)
        rest_half = cv2.resize(rhalf, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, rest_half)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def setquarterrest(hpos, vpos) :
        
        Xsize = 20
        Ysize = 60

        rquarter = cv2.imread(imgpath + "quarterrest.png", cv2.IMREAD_GRAYSCALE)
        rest_quarter = cv2.resize(rquarter, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, rest_quarter)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set8threst(hpos, vpos) :
        
        Xsize = 30
        Ysize = 40

        r8th = cv2.imread(imgpath + "8threst.png", cv2.IMREAD_GRAYSCALE)
        rest_8th = cv2.resize(r8th, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, rest_8th)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def set16threst(hpos, vpos) :
        
        Xsize = 30
        Ysize = 40

        r16th = cv2.imread(imgpath + "16threst.png", cv2.IMREAD_GRAYSCALE)
        rest_16th = cv2.resize(r16th, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, rest_16th)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST

def addrestdot(hpos, vpos) :
        
        Xsize = 4
        Ysize = 4

        d = cv2.imread(imgpath + "dot.png", cv2.IMREAD_GRAYSCALE)
        dot = cv2.resize(d, dsize=(Xsize,Ysize), interpolation=cv2.INTER_AREA)

        #rows, cols, channels = wholenote.shape
        ROI = score[vpos : Ysize + vpos, hpos : Xsize + hpos]
        DST = cv2.bitwise_or(ROI, dot)

        score[vpos : Ysize + vpos, hpos : Xsize + hpos] = DST


##
note_start = np.array([])
n = 0

note_start = np.hstack([note_start, n])

sample_note_start = np.array([0,16,32, 64,80,96, 128,144,160,176,192,208,224, 256,272,288,304,320,336,352, 384,400,416,432,448,464,496])
sample_note_pitch = np.array([46,43,43, 44,41,41, 39,41,43,44,46,46,46, 46,43,43,43,44,41,41, 39,43,46,46,43,43,43])
sample_note_time = np.array([16,16,32, 16,16,32, 16,16,16,16,16,16,32, 16,16,16,16,16,16,32, 16,16,16,16,16,16,32])

start = sample_note_start
pitch = sample_note_pitch
time = sample_note_time

for n in range(len(Q_f0)) :
        n = n + 1

        if time[n] == 16 :
                
