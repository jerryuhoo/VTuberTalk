# VTuberTalk

## 0. 介绍

这是一个根据VTuber的声音训练而成的TTS（text-to-speech）模型，输入文本和VTuber可以输出对应的语音。本项目基于[百度PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)。

## 1. 环境安装 && 准备

python >= 3.8

参考[paddlepaddle安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

GPU安装:

```shell
python -m pip install paddlepaddle-gpu==2.2.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

pip install paddlespeech
```

CPU安装：

```shell
pip install paddlepaddle paddlespeech
```

目录结构：

```
├── train
├── gui
├── tools
├── MFA
│   ├── mandarin_pinyin.dict
│   └── mandarin.zip
└── data
    ├── wav
    │   ├── speaker_name1
    │   │   ├── video
    │   │   ├── raw
    │   │   ├── unrecognized            
    │   │   └── split   
    │   │       ├── .wav 
    │   │       ├── .txt
    │   │       └── .lab 
    │   └── speaker_name2
    ├── TextGrid
    │   ├── speaker_name1
    │   │   └── .TextGrid
    │   └── speaker_name2
    └── durations.txt
```

## 2. 预处理

### 2.1. 从直播录像中获取音频

从B站获取音频的方法：
可以用bilibili助手下载Vtuber的录播flv文件，再转成wav文件。

从YouTube获取音频的方法：
可以用TamperMonkey上的YouTube下载器下载mp4文件，再转成wav文件。

安装依赖库：

```shell
pip install pydub
```

```shell
python tools/video_to_wav.py --path <data to folder or file>
```

其中，在video_to_wav可设置采样率，一般设置为16000，因为如果要使用语音切分工具的话，16000是支持的采样率之一。

### 2.2. 将音频分割成片段

步骤2.2和2.3仅限于没有字幕的音频，如果在YouTube下载的话大概率会有字幕文件，下载字幕文件后直接跳转到“2.4. 使用字幕获得文本”即可。

音频分割使用了webrtcvad模块，其中第一个参数aggressiveness是分割检测的敏感度，数字越大，对于静音检测越敏感，分割的音频个数也越多。范围为0～3。

```shell
python tools/split_audio.py --ag <aggressiveness> --in_path <data/wav/speaker_name/raw>
```

### 2.3. 使用ASR获得文本

```shell
python gen_text.py --path <data/wav/speaker_name/split> --lang <language: 'en' or 'zh'>
```

### 2.4. 使用字幕获得文本

文件夹中可以有多个wav和srt文件，对应的wav和srt需要同名。

```shell
python tools/split_audio_by_srt.py --path <data>
```

### 2.5. 去除过长过短文本

```shell
python tools/data_filter.py --path <data/wav/speaker_name/split>
```

### 2.6. 汉字转拼音

```shell
python tools/hanzi_to_pinyin.py --path <data/wav/speaker_name/split>
```

### 2.7. Spleeter降噪

WIP

```shell
pip install spleeter
```

### 2.8. MFA音素对齐

本项目使用了百度PaddleSpeech的fastspeech2模块作为tts声学模型。

安装MFA

```shell
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

下载[mandarin](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/models/acoustic.html)模型，放入MFA文件夹中。

```shell
mfa align <data/wav/speaker_name/split> MFA/mandarin_pinyin.dict MFA/mandarin.zip <data/TextGrid/speaker_name>
```

> 如果再使用需要加`--clean`

> 如果要生成MFA1.x版本（包含sp和sil信息）需要加`--disable_textgrid_cleanup True`

### 2.9. 生成其他预处理文件

生成duration

```shell
python tools/gen_duration_from_textgrid.py \
    --inputdir=data/TextGrid \
    --output=data/durations.txt \
    --config=train/conf/default_multi.yaml
```

提取features

```shell
python train/preprocess.py \
    --dataset=other \
    --rootdir=data/ \
    --dumpdir=dump \
    --dur-file=data/durations.txt \
    --config=train/conf/default_multi.yaml \
    --num-cpu=16 \
    --cut-sil=True
```

compute_statistics

```shell
python tools/compute_statistics.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --field-name="speech"

python tools/compute_statistics.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --field-name="pitch"

python tools/compute_statistics.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --field-name="energy"
```

normalize

```shell
python tools/normalize.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --dumpdir=dump/train/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt

python tools/normalize.py \
    --metadata=dump/dev/raw/metadata.jsonl \
    --dumpdir=dump/dev/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt

python tools/normalize.py \
    --metadata=dump/test/raw/metadata.jsonl \
    --dumpdir=dump/test/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
```

## 3. 训练

```shell
python train/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=train/conf/default_multi.yaml \
    --output-dir=exp/fastspeech2_bili3_aishell3 \
    --ngpu=1 \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
```

## 4. 推理

下载[pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip)。

下载[hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip)。

### 1. fastspeech + pwg + multiple

```shell
python train/synthesize_e2e.py \
        --am=fastspeech2_aishell3 \
        --am_config=train/conf/default_multi.yaml \
        --am_ckpt=exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_<iter num>.pdz \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_csmsc \
        --voc_config=pwg_baker_ckpt_0.4/pwg_default.yaml \
        --voc_ckpt=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --voc_stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --ngpu=1 \
        --spk_id=218
```

### 2. fastspeech + hifigan + single

```shell
python train/synthesize_e2e.py \
        --am=fastspeech2_csmsc \
        --am_config=train/conf/default_single.yaml \
        --am_ckpt=exp/fastspeech2_ghost/checkpoints/snapshot_iter_<iter num>.pdz \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --voc_config=hifigan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=dump/phone_id_map.txt \
        --ngpu=1
```

## 5. GUI界面 (WIP)

安装依赖库：

```shell
pip install PyQt5
pip install sounddevice
```

启动GUI界面：

```shell
cd gui/
python main2.py
```
