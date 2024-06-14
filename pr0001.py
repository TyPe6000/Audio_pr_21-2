# pr 넘버링은 실습을 위한 코드입니다. prex넘버링은 연습용/조사용입니다.
# 물론 미완성입니다. 역시 미숙한 부분이 많습니다.
# python 3.8.6 버전을 활용해 windows10 OS, VisualStudioCode 환경에서 코딩하는 내용입니다.
# VSCode 설치 및 python 환경 구축에 관해선 다음의 링크 참고 https://nadocoding.tistory.com/4

# 주로 참고한 내용 : librosa, wavfile, numpy/scipy로 푸리에 변환 등
# https://data-science-note.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-wav-%ED%8C%8C%EC%9D%BC-%EC%9D%BD%EA%B8%B0
# https://jjeongil.tistory.com/654
# https://wikidocs.net/92071
# https://coding-yoon.tistory.com/64
# https://codetorial.net/matplotlib/subplot.html
# https://ballentain.tistory.com/3

# python에 library를 설치해야합니다.
# pip install "설치할 라이브러리"
# numpy scipy.io matplotlib librosa(현시점 미사용)
# 정리가 덜되어서 난잡합니다. 
import numpy as np 
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt # 그래프 표현 함수 관련 lib
from numpy import arange, ones, pi, cos, sin
from scipy.fftpack import fft, fftfreq, ifft

# 주소는 역슬래시가 아니라 슬래시로 적어야 구동됩니다.
filepath = "C:/code/audio_pr/sound/"
filename = "sample0001.wav"

# scipy에 포함된 wavfile 활용, wav 형식 파일을 읽어온다.
# samplerate : 샘플링주파수, data : 위상
samplerate, data = wavfile.read(filepath + filename)

# arange는 numpy.arange([start, ] stop, [step, ] dtype=None) 에서, start부터  stop까지 step 간격으로 array 반환
# len() : len(list)에서 list의 크기(list의 데이터 개수)
# data데이터 개수만큼 간격이 1인 array를 만들고, 그걸 samplerate로 나눈다.
# 따라서 times는 1/samplerate초 단위, data개수만큼의 크기의 시간 array
#sd.play(data, samplerate)
# 읽어들인 data, samplerate를 활용해서 음성 재생                     
times = np.arange(len(data))/float(samplerate)

# --분석 덜된 부분입니다.-- 푸리에 변환 fft
fft_data = fft(data)

# matplotlib.pyplot 활용해 그래프 작성
plt.fill_between(times, fft_data)
plt.xlabel("time")
plt.ylabel("fft_data")
plt.xlim()
plt.ylim()

plt.show()

#0.05초 기준으로 시간 분할
unittime = 0.05
unitsample = samplerate*unittime


