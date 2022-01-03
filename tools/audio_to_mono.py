from pydub import AudioSegment
import os
import argparse

def process(args):
    path=args.path
    files=os.listdir(path)
    for file in files:
        sound = AudioSegment.from_wav(os.path.join(args.path, file))
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(int(args.sr))
        sound.export(os.path.join(args.path, file), format="wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--sr", type=str, default=16000)
    args = parser.parse_args()
    process(args)