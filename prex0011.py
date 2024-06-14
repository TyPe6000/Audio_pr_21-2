
import librosa
import matplotlib.pyplot as plt
import librosa.display

filepath = "C:/code/audio_pr/sound/"
filename = "sample0001.wav"



## x - time, sr - samplingrate
x , sr = librosa.load(filepath + filename, sr=48000) 

#display waveform

#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(x, sr=sr)

## display Spectrogram
X = librosa.stft(x, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=None, pad_mode='reflect')
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))
librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='hz') 
## If to print log of frequencies  
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()


#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to print log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#plt.colorbar()
#plt.show()

