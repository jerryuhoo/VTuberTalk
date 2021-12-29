import os
import argparse

def move(in_path, out_path):
    os.replace(in_path, out_path)

def process(args, outdir):
    path=args.path
    files=os.listdir(path)
    for folder in files:
        if not os.path.isdir(os.path.join(args.path, folder)):
            continue
        else:
            out_name = folder + ".wav"
            move(os.path.join(args.path, folder, "vocals.wav"), os.path.join(outdir, out_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="../clean_raw2")
    args = parser.parse_args()
    
    outdir = os.path.join(args.path, args.out_path)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    process(args, outdir)

    print("you can delete clean_raw folder now.")

