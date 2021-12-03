"""
change wav sample rate
"""

import os
import librosa
import soundfile as sf
sample_rate_before = 48000
sample_rate_after = 16000

g = os.walk(r"./")

for path, dir_list, file_list in g:
    for file in file_list:
        if file.endswith('.wav'):
            print("resampling: ", os.path.join(path, file))
            y, sr = librosa.load(file, sr=sample_rate_before)
            y_out = librosa.resample(y, sr, sample_rate_after)
            sf.write(file, y_out, sample_rate_after, subtype='PCM_24')