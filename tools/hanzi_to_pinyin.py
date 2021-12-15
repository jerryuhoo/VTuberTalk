from pypinyin import pinyin, Style
import os
import argparse

def get_pinyin_from_text(text):
    pinyins = [
        p[0] for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
    ]
    return pinyins

def process(path):
    is_dir = os.path.isdir(path)
    if is_dir: 
        path_list=os.listdir(path)
    else: # input is a file
        path, basename = os.path.split(path)
        path_list = [basename]

    for filename in path_list:
        if os.path.isdir(os.path.join(path, filename)):
            continue
        filename_suffix = os.path.splitext(filename)[1]
        print(filename)
        input_file_path = os.path.join(path, filename)
        output_file_path = os.path.join(path, os.path.splitext(filename)[0] + ".lab")
        if filename_suffix == '.txt':
            with open(input_file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        # print(text.encode(encoding="utf-8"))
                        pinyins = get_pinyin_from_text(text)
            f.close()
            with open(
                    os.path.join(output_file_path),
                    "w",
                ) as f1:
                    for pinyin in pinyins:
                        f1.write("%s " % pinyin)
            f1.close()
            print("write to " + output_file_path)
        else:
            print("file ", filename, " format not supported!")
            continue
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    is_exist = os.path.exists(args.path)
    if not is_exist:
        print("path not existed!")
    else:
        process(args.path)