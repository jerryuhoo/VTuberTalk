from pydub import AudioSegment
sound = AudioSegment.from_wav("2021_7_22_23_in.wav")
sound = sound.set_channels(1)
sound = sound.set_frame_rate(16000)
sound.export("2021_7_22_23.wav", format="wav")