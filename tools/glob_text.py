import os
import re
import argparse

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def process(files, path):
    files = sorted_alphanumeric(files)
    for file in files:
        if file.endswith('.txt'):
            pass
        else:
            continue
        position = path + file
        print(position)
        with open(position ,'r') as f:
            for line in f.readlines():
                with open("./text.txt","a") as p:
                    p.write(str(file) + " " + line + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    files=os.listdir(args.path)

    if os.path.exists("./text.txt"):
        os.remove("./text.txt")

    process(files, args.path)