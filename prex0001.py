# prex 넘버링은 정보 수집과정, 테스트를 위해 무작정 만들고 돌리고 수정하는 시험용 프로그램 집합체입니다.
# 크게 의미 없는 정보에 주의 바람.

# wavfile 함수 기초, wav형식 음원 불러와서 시간 축, 위상 축 2차원 그래프 그리는 프로그램

# https://data-science-note.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-wav-%ED%8C%8C%EC%9D%BC-%EC%9D%BD%EA%B8%B0
# https://jjeongil.tistory.com/654
# https://wikidocs.net/92071
# https://coding-yoon.tistory.com/64
# https://codetorial.net/matplotlib/subplot.html
# https://ballentain.tistory.com/3

import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt

import sounddevice as sd

path = "C:/__file/pcd/audio_pr/sound/"
filename = "sample0001.wav"

samplerate, data = wavfile.read(path + filename)
#wavfile.read 로 wav음원파일을 read
# samplerate : 샘플링주파수, data : 위상

times = np.arange(len(data))/float(samplerate)
# arange는 numpy.arange([start, ] stop, [step, ] dtype=None) 에서, start부터  stop까지 step 간격으로 array 반환
# len() : len(list)에서 list의 크기(list의 데이터 개수)
# data데이터 개수만큼 간격이 1인 array를 만들고, 그걸 samplerate로 나눈다.
# 따라서 times는 1/samplerate초 단위, data개수만큼의 크기의 시간 array
#sd.play(data, samplerate)
# 읽어들인 data, samplerate를 활용해서 음성 재생

print(samplerate, data.shape)
print(data)
print(times)

plt.fill_between(times, data)
plt.xlabel("sample")
plt.ylabel("data")
plt.xlim(times[0], times[-1])
plt.grid()
plt.show()

print( ' sampling rate : ', samplerate)
print('time : ', times[-1])
