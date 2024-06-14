import librosa
import numpy as np
import os.path
import matplotlib.pyplot as plt

wav = "C:/__file/pcd/audio_pr/sound/sample0001.wav"

(file_dir, file_id) = os.path.split(wav)
print("file_dir:", file_dir)
print("file_id:", file_id)

y, sr = librosa.load(wav, sr=48000)
time = np.linspace(0, len(y)/sr, len(y)) #time axis

fig, ax1 = plt.subplots()
ax1.plot(time, y, color = 'b', label='waveform')
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time[s]")
plt.title(file_id) # 제목
plt.savefig(file_id+'.png')
plt.show()