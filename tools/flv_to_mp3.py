from pydub import AudioSegment
import os

path = './'
path_list=os.listdir(path)
for filename in path_list:
    if os.path.splitext(filename)[1] == '.flv':
        print(filename)
        sound = AudioSegment.from_flv(filename)
        sound = sound.set_frame_rate(22050)
        sound.export(os.path.splitext(filename)[0] + ".wav", format="wav")