from pydub import AudioSegment
import os
import argparse

def process(path):
    path_list=os.listdir(path)
    for filename in path_list:
        if os.path.isdir(os.path.join(path, filename)):
            continue
        filename_suffix = os.path.splitext(filename)[1]
        input_file_path = os.path.join(path, filename)
        output_wav_path = os.path.join(path, os.path.splitext(filename)[0])
        output_txt_path = os.path.join(path, os.path.splitext(filename)[0])
        srt_list = []
        if filename_suffix == '.srt':
            with open(input_file_path, 'r', encoding="UTF-8") as f:
                srt_text = f.read()
                for sentence_text in srt_text.split("\n\n"):
                    sentence_list = sentence_text.split("\n")
                    if sentence_list != ['']:
                        srt_list.append(sentence_list)
            wav_path = os.path.join(path, os.path.splitext(filename)[0] + ".wav")
            if(os.path.isfile(wav_path)):
                wav = AudioSegment.from_wav(wav_path)
                cut_audio(wav, srt_list, output_wav_path, output_txt_path)

def srt_time_to_mstime(time_str):
    l = time_str.split(" --> ")
    start_time = l[0]
    end_time = l[1]
    start_time_l = start_time.split(",")
    h, m, s = start_time_l[0].split(":")
    start_time = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(start_time_l[1])
    end_time_l = end_time.split(",")
    h, m, s = end_time_l[0].split(":")
    end_time = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(end_time_l[1])
    return start_time, end_time

def cut_audio(wav, srt_list, output_wav_path, output_txt_path, bias=50): # add bias to fit different videos
    for i, sentence_list in enumerate(srt_list):
        sentence_time = sentence_list[1]
        sentence_text = sentence_list[2]
        print(sentence_text)
        start_time, end_time = srt_time_to_mstime(sentence_time)
        print(start_time)
        print(end_time)
        end_time = min(end_time, len(wav))
        sentence_wav = wav[start_time + bias:end_time + bias]
        sentence_wav.export(output_wav_path + "_" + str(i) + ".wav", format="wav")
        with open(output_txt_path + "_" + str(i) + ".txt", 'w', encoding="UTF-8") as f:
            f.write(sentence_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    
    is_exist = os.path.exists(args.path)
    if not is_exist:
        print("path not existed!")
    else:
        path = args.path
        process(path)
