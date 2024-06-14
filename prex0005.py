#librosa.load로 데이터 가져오기 테스트

import librosa
import matplotlib.pyplot as plt

wav_file = "C:/__file/pcd/audio_pr/sound/sample0001.wav"

wavdata, samplerate = librosa.load(wav_file, sr=None)

print("wavdata.shape : ", wavdata.shape)
print("wavdata[0].type : ", type(wavdata[0]))
print("samplerate : ", samplerate)

plt.fill_between(samplerate, wavdata)
plt.show()
