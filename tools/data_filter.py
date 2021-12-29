from genericpath import exists
import os
import re
import argparse

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def move(root_path, file_name, output_path):
    os.replace(os.path.join(root_path, file_name), os.path.join(output_path, file_name))
    wav_file = os.path.splitext(file_name)[0] + '.wav'
    os.replace(os.path.join(root_path, wav_file), os.path.join(output_path, wav_file))

def process(args, outdir):
    path=args.path
    files=os.listdir(path)
    files = sorted_alphanumeric(files)
    count = [0] * 200
    for file in files:
        if file.endswith('.txt'):
            pass
        else:
            continue
        position = path + file
        with open(position ,'r') as f:
            for line in f.readlines():
                line_len = len(line)
                count[line_len] += 1
                if line_len < int(args.min) or line_len > int(args.max):
                    move(path, file, outdir)

    for i in range(len(count)):
        print("长度为", i, "的有", count[i], "条。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--min", type=str, default=1)
    parser.add_argument("--max", type=str, default=50)
    args = parser.parse_args()
    
    outdir = os.path.join(args.path, "../useless")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    process(args, outdir)

