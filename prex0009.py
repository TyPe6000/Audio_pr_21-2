# fft 예시 프로그램

## 1. Import modules
import matplotlib.pyplot as plt
from numpy import arange, ones, pi, cos, sin
from scipy.fftpack import fft, fftfreq, ifft


## 2. Set up for domain & Creating signal
n   = 100          # 한 파장당 지점 개수
Lx  = 10           # 한 파장의 거리 (혹은 시간)
L   = Lx/(2*pi)    # 파장
omg = 2.0*pi/Lx    # 각진동수

x   = arange(0, n)/Lx       # x축 : n개의 지점, Lx의 길이를 한 파장으로 가정
y1  = 5.0*sin( 2.0*omg*x)   # 파수  2에 해당하는 사인파
y2  = 4.0*cos( 3.0*omg*x)   # 파수  3에 해당하는 코사인파
y3  = 1.0*sin(10.0*omg*x)   # 파수 10에 해당하는 사인파
y   = y1 + y2 + y3 # + 1    # 임의의 파동 y (상수값을 추가하려면 주석을 해제할 것)


## 3. Preparatory steps
freqs = fftfreq(n)    # 필요한 모든 진동수를 만든다.
mask = freqs > 0      # 절반의 값을 무시
nwaves = freqs*n      # 도메인 길이에 따른 파수


## 4. FFT calculations
fft_vals = fft(y)               # FFT 계산
fft_norm = fft_vals*(1.0/n)     # FFT 계산된 결과를 정규화
fft_theo = 2.0*abs(fft_norm)    # 푸리에 계수 계산

# 계산하고싶은 파수의 범위를 지정 (0~50 사이의 숫자를 입력)
wavenumber = int(input("input wavenumber (~50) : ",))

x0  = ones(n)
origin = fft_norm.real[0]*x0    # 상수부분인 푸리에 계수를 a0 더함

for k in range(1, wavenumber+1):    # 푸리에계수 an, bn을 이용해 IFFT 구현
    origin +=   2 * fft_norm.real[k] * cos(k*omg*x) + \
              (-2)* fft_norm.imag[k] * sin(k*omg*x)


## 5. Check result
# 1. 임의의 파동 y
plt.figure()
plt.subplot(211)
plt.plot(x, y, color='k', label='Original')
plt.plot(x, origin, color='g', label='IFFT')
plt.title("Original & IFFT Signal")
plt.legend()

# 2. 푸리에 계수
plt.subplot(212)
plt.bar(freqs[mask]*n, fft_theo[mask], label="true fft values")
plt.title("True FFT values")
plt.axhline(y=0, linewidth=1, color='k')
plt.legend()
plt.show()