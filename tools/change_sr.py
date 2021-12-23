"""
change wav sample rate
"""
import argparse
import os
import librosa
import soundfile as sf
import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Change audio sample rate.")
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="input folder path")
    parser.add_argument(
        "--sr", type=str, required=True, help="output sample rate.")
    args = parser.parse_args()

    g = os.walk(args.path)
    sample_rate = int(args.sr)
    
    for path, dir_list, file_list in g:
        for file in tqdm.tqdm(file_list, total=len(file_list)):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(args.path, file)
                # print("resampling: ", file_path)
                y, sr = librosa.load(file_path, sr=sample_rate)
                if file.endswith('.mp3'):
                    file_path = os.path.join(args.path, os.path.splitext(file)[0] + ".wav")
                sf.write(file_path, y, sample_rate, subtype='PCM_16')
    print("change sample rate to ", sample_rate, " , Done.")

if __name__ == '__main__':
    main()