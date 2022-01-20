import os
import re
import argparse

def move(root_path, file_name, output_path):
    os.replace(os.path.join(root_path, file_name), os.path.join(output_path, file_name))
    wav_file = os.path.splitext(file_name)[0] + '.wav'
    lab_file = os.path.splitext(file_name)[0] + '.lab'
    os.replace(os.path.join(root_path, wav_file), os.path.join(output_path, wav_file))
    if os.path.exists(os.path.join(root_path, lab_file)):
        os.replace(os.path.join(root_path, lab_file), os.path.join(output_path, lab_file))

def process(files, path):
    text_dict = {}
    with open("./text.txt" ,'r', encoding='utf-8') as text_file:
        for line in text_file.readlines():
            line = line[:-1]
            file_name, text = line.split()[0], re.search(r'[".txt "].*',line).group()[5:]
            text_dict[file_name] = text
    text_file.close()
    for file in files:
        if not file.endswith('.txt'):
            continue
        position = os.path.join(path, file)
        with open(position ,'r', encoding='utf-8') as f:
            ori_text = ""
            for line in f.readlines():
                ori_text = line
            if file in text_dict.keys():
                revised_text = text_dict[file]
            else:
                outdir = os.path.join(path, "../unrecognized")
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                f.close()
                move(path, file, outdir)
                print("move " + file + " to unrecognized.")
                continue

            if revised_text != ori_text:
                print(str(file) + " " + ori_text + " --> " + revised_text)
                with open(position, 'w', encoding='utf-8') as revised_file:
                    revised_file.write(revised_text)
                revised_file.close()
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    files=os.listdir(args.path)

    if not os.path.exists("./text.txt"):
        print("no text.txt found!")
    else:
        process(files, args.path)