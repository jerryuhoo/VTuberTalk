from pydub import AudioSegment
import os
import argparse

def process(path, output_sample_rate):
    path_list=os.listdir(path)
    print(path)
    for filename in path_list:
        filename_suffix = os.path.splitext(filename)[1]
        print(filename)
        input_file_path = os.path.join(path, filename)
        output_file_path = os.path.join(path, os.path.splitext(filename)[0] + ".wav")
        if filename_suffix == '.flv':
            sound = AudioSegment.from_flv(input_file_path)
            sound = sound.set_frame_rate(output_sample_rate)
            sound.export(os.path.join(output_file_path), format="wav")
        elif filename_suffix == '.mp4':
            cmd = "ffmpeg -i {} -ac 1 -ar {} -f wav {}".format(input_file_path, output_sample_rate, output_file_path)
            os.system(cmd)
        else:
            print("file ", filename, " format not supported!")
            continue
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    output_sample_rate = 22050
    is_exist = os.path.exists(args.path)
    if not is_exist:
        print("path not existed!")
    else:
        path = args.path
        process(path, output_sample_rate)