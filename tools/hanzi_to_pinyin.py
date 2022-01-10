from pypinyin import pinyin, Style
import os
import argparse
import re

def get_pinyin_from_text(text):
    pinyins = [
        p[0] for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
    ]
    return pinyins

def process(path, ignore_punc):
    is_dir = os.path.isdir(path)
    if is_dir: 
        path_list=os.listdir(path)
    else: # input is a file
        path, basename = os.path.split(path)
        path_list = [basename]

    for filename in path_list:
        root_folder = os.path.join(path, filename)
        if os.path.isdir(root_folder):
            process(root_folder, ignore_punc)
        filename_suffix = os.path.splitext(filename)[1]
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
                        if ignore_punc:
                            rule = re.compile(u"[^a-zA-Z0-9]")
                            pinyin = rule.sub('', pinyin)
                            if pinyin == '':
                                continue
                        f1.write("%s " % pinyin)
            f1.close()
            # print("write to " + output_file_path)
        else:
            # print("file ", filename, " format not supported!")
            continue

def str2bool(str):
    return True if str.lower() == 'true' else False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ignore_punc", type=str2bool, default=True)
    args = parser.parse_args()

    is_exist = os.path.exists(args.path)
    if not is_exist:
        print("path not existed!")
    else:
        process(args.path, args.ignore_punc)

    print("done!")