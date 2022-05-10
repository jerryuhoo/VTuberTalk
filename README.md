# VTuberTalk

## 0. 介绍

这是一个根据VTuber的声音训练而成的TTS（text-to-speech）模型，输入文本和VTuber可以输出对应的语音。本项目基于[百度PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)。

Demo视频：
<div align="center"><a href="https://www.bilibili.com/video/BV1Wa41167fX"><img src="https://i1.hdslb.com/bfs/archive/78f4321b480694bcb00b9f542c828ab66b3b4ee9.jpg@640w_400h_100Q_1c.webp" / width="500px"></a></div>

## 1. 环境安装 && 准备

### 1.1. 安装ffmepg

Windows:
首先检查一下自己有没有安装过ffmpeg，如果没有就下载
[ffmpeg](https://ffmpeg.org/download.html)

[参考教程](https://zhuanlan.zhihu.com/p/118362010)

Mac：

```basj
brew install ffmpeg
```

Ubuntu：

```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```

### 1.2. conda搭建环境

python >= 3.8

gpu版本：建议训练模型使用这个版本，需要先配置好cuda环境。

```bash
conda create -n paddlespeech python=3.8
conda activate paddlespeech
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

cpu版本：如果只是单纯使用，建议安装这个版本。

```bash
conda create -n paddlespeech python=3.8
conda activate paddlespeech
pip install -r requirements_cpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3. 手动安装cpu/gpu版本的paddlepaddle（如果正确安装则跳过）

参考[paddlepaddle安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

【需要paddle 2.3.0以上版本】

### 1.4. 目录结构

```text
├── train
├── gui
├── tools
├── pretrained_models
│   ├── 2stems
│   ├── pwg_aishell3_ckpt_0.5
│   └── hifigan_csmsc_ckpt_0.1.1
├── MFA
│   ├── pinyin_eng.dict
│   └── mfa_model.zip
└── data
    ├── wav_temp
    │   ├── speaker_name1
    │   │   ├── video
    │   │   ├── raw
    │   │   ├── clean_raw2
    │   │   ├── unrecognized
    │   │   ├── unused          
    │   │   └── split   
    │   │       ├── .wav 
    │   │       ├── .txt
    │   │       └── .lab 
    │   └── speaker_name2
    ├── TextGrid_temp
    │   ├── speaker_name1
    │   │   └── .TextGrid
    │   └── speaker_name2
    ├── wav
    │   ├── speaker_name1
    │   │   ├── .wav 
    │   │   ├── .txt
    │   │   └── .lab 
    │   └── speaker_name2
    ├── TextGrid
    │   ├── speaker_name1
    │   │   └── .TextGrid
    │   └── speaker_name2
    └── durations.txt
```

## 2. 数据准备

### 2.0. 一键处理（包含2.1到2.9）

如果运行这一步则可以忽略2.1-2.9，只需要把你的视频文件（flv格式）放在data/wav_temp/speaker_name/video文件夹中即可。
在run_preprocess.sh中指定你想要的stage，建议在stage 5之后手动修正错误的语音识别结果。

```shell
./run_preprocess.sh
```

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

可选项：如果视频过长，使用以下的命令将视频切割

```shell
python tools/cut_source.py --path <data/wav/video/> --min <minute to cut> --sr <sample rate>
```

其中，在video_to_wav可设置采样率，一般设置为16000，因为如果要使用语音切分工具的话，16000是支持的采样率之一。

### 2.2. Spleeter降噪

直接运行可能会因为网络的原因下载模型失败，建议直接先下载好[模型](https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz)，放到pretrained_models/2stems中。

```shell
pip install spleeter
spleeter separate \
     -o <data/wav/speaker_name/clean_raw/> \
     <data/wav/speaker_name/raw/*.wav>
```

> 如果遇到CUDA的报错试试执行`export TF_FORCE_GPU_ALLOW_GROWTH=true`

获取降噪后的人声并且重命名，这步做完之后的文件在clean_raw2，可以删除clean_raw。

```shell
python tools/glob_spleeter_vocals.py --path <data/wav/speaker_name/clean_raw/>
```

降噪后又变成了双声道，因此需要执行

```shell
python tools/audio_to_mono.py --path <data/wav/speaker_name/clean_raw2/>
```

### 2.3. 将音频分割成片段

步骤2.2和2.3仅限于没有字幕的音频，如果在YouTube下载的话大概率会有字幕文件，下载字幕文件后直接跳转到“2.4. 使用字幕获得文本”即可。

音频分割使用了webrtcvad模块，其中第一个参数aggressiveness是分割检测的敏感度，数字越大，对于静音检测越敏感，分割的音频个数也越多。范围为0～3。

```shell
python tools/split_audio.py --ag <aggressiveness> --in_path <data/wav/speaker_name/clean_raw2/>
```

### 2.4. 使用ASR获得文本

```shell
python tools/gen_text.py --path <data/wav/speaker_name/split/> --lang <language: 'en' or 'zh'>
```

### 2.5. 使用字幕获得文本

文件夹中可以有多个wav和srt文件，对应的wav和srt需要同名。

```shell
python tools/split_audio_by_srt.py --path <data>
```

### 2.6. 去除过长过短文本

```shell
python tools/data_filter.py --path <data/wav/speaker_name/split/>
```

### 2.7. 文本纠正

收集所有的文本到一个txt文件中。

```shell
python tools/glob_text.py --path <data/wav/speaker_name/split/>
```

打开txt文件，修改错字后再运行

```shell
python tools/revise_text.py --path <data/wav/speaker_name/split/>
```

### 2.8. 汉字转拼音

```shell
python tools/hanzi_to_pinyin.py --path <data/wav/speaker_name/split/>
```

### 2.9. MFA音素对齐

本项目使用了百度PaddleSpeech的fastspeech2模块作为tts声学模型。

安装MFA

```shell
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

自己训练一个，详见[MFA训练教程](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-train-acoustic-model)

单人数据集：

```shell
python tools/generate_lexicon.py pinyin --with-r --with-tone
mfa train <data/wav/speaker_name/split/> MFA/pinyin.dict MFA/mandarin.zip <data/TextGrid/speaker_name/>
```

多人数据集：

```shell
python tools/generate_lexicon.py pinyin --with-r --with-tone
mfa train <data/wav/> MFA/pinyin.dict MFA/mandarin.zip <data/TextGrid/>
```

（可选）如果已经有MFA模型了可以执行这一步以节约时间，但还是建议从头开始训练。

```shell
mfa align <data/wav/speaker_name/split/> MFA/pinyin.dict MFA/mandarin.zip <data/TextGrid/speaker_name/>
```

> 如果再使用需要加`--clean`

> 如果要生成MFA1.x版本（包含sp和sil信息）需要加`--disable_textgrid_cleanup True`

### 2.10. 生成其他预处理文件

#### 一键运行

```shell
./run_train.sh
```

#### 生成duration

##### 1. fastspeech2 模型（多人）

```shell
python tools/gen_duration_from_textgrid.py \
    --inputdir=data/TextGrid \
    --output=data/durations.txt \
    --config=train/conf/fastspeech2/default_multi.yaml
```

##### 2. speedyspeech 模型

单人

```shell
python tools/gen_duration_from_textgrid.py \
    --inputdir=data/TextGrid \
    --output=data/durations.txt \
    --config=train/conf/speedyspeech/default.yaml
```

多人

```shell
python tools/gen_duration_from_textgrid.py \
    --inputdir=data/TextGrid \
    --output=data/durations.txt \
    --config=train/conf/speedyspeech/default_multi.yaml
```

#### 提取features

##### 1. fastspeech2 模型（多人）

```shell
python train/exps/fastspeech2/preprocess.py \
    --dataset=other \
    --rootdir=data/ \
    --dumpdir=dump \
    --dur-file=data/durations.txt \
    --config=train/conf/fastspeech2/default_multi.yaml \
    --num-cpu=16 \
    --cut-sil=True
```

##### 2. speedyspeech 模型

单人

```shell
python train/exps/speedyspeech/preprocess.py \
    --dataset=other \
    --rootdir=data/ \
    --dumpdir=dump \
    --dur-file=data/durations.txt \
    --config=train/conf/speedyspeech/default.yaml \
    --num-cpu=16 \
    --cut-sil=True \
    --use-relative-path=True
```

多人

```shell
python train/exps/speedyspeech/preprocess.py \
    --dataset=other \
    --rootdir=data/ \
    --dumpdir=dump \
    --dur-file=data/durations.txt \
    --config=train/conf/speedyspeech/default_multi.yaml \
    --num-cpu=16 \
    --cut-sil=True \
    --use-relative-path=True
```

#### compute_statistics

##### 1. fastspeech2 模型

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

##### 2. speedyspeech 模型

```shell
python tools/compute_statistics.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --field-name="feats" \
    --use-relative-path=True
```

#### normalize

如果是在已有模型上替换speaker进行finetune，在这一步之前需要将生成的phone_id_map.txt替换为已有模型的音素词典，不然phone ip映射错误，对训练的发音产生影响。

##### 1. fastspeech2 模型

```shell
python train/exps/fastspeech2/normalize.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --dumpdir=dump/train/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt

python train/exps/fastspeech2/normalize.py \
    --metadata=dump/dev/raw/metadata.jsonl \
    --dumpdir=dump/dev/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt

python train/exps/fastspeech2/normalize.py \
    --metadata=dump/test/raw/metadata.jsonl \
    --dumpdir=dump/test/norm \
    --speech-stats=dump/train/speech_stats.npy \
    --pitch-stats=dump/train/pitch_stats.npy \
    --energy-stats=dump/train/energy_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
```

##### 2. speedyspeech 模型

单人

```shell
python train/exps/speedyspeech/normalize.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --dumpdir=dump/train/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --use-relative-path=True

python train/exps/speedyspeech/normalize.py \
    --metadata=dump/dev/raw/metadata.jsonl \
    --dumpdir=dump/dev/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --use-relative-path=True

python train/exps/speedyspeech/normalize.py \
    --metadata=dump/test/raw/metadata.jsonl \
    --dumpdir=dump/test/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --use-relative-path=True
```

多人

```shell
python train/exps/speedyspeech/normalize.py \
    --metadata=dump/train/raw/metadata.jsonl \
    --dumpdir=dump/train/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt \
    --use-relative-path=True

python train/exps/speedyspeech/normalize.py \
    --metadata=dump/dev/raw/metadata.jsonl \
    --dumpdir=dump/dev/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt \
    --use-relative-path=True

python train/exps/speedyspeech/normalize.py \
    --metadata=dump/test/raw/metadata.jsonl \
    --dumpdir=dump/test/norm \
    --stats=dump/train/feats_stats.npy \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt \
    --use-relative-path=True
```

## 3. 训练

### 3.1. fastspeech2 模型（多人）

```shell
python train/exps/fastspeech2/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=train/conf/fastspeech2/default_multi.yaml \
    --output-dir=exp/fastspeech2_bili3_aishell3 \
    --ngpu=1 \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
```

### 3.2. speedyspeech模型

单人

```shell
python train/exps/speedyspeech/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=train/conf/speedyspeech/default.yaml \
    --output-dir=exp/speedyspeech_bili3_aishell3 \
    --ngpu=1 \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --use-relative-path=True
```

多人

```shell
python train/exps/speedyspeech/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=train/conf/speedyspeech/default_multi.yaml \
    --output-dir=exp/speedyspeech_azi_nanami \
    --ngpu=1 \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt \
    --use-relative-path=True
```

### 3.3. 查看Loss图

```shell
visualdl --logdir <log folder path>
```

## 4. 推理/导出静态模型

下载[pwg_aishell3_ckpt_0.5](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip)。

下载[hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip)。

把下载的vocoder模型放在pretrained_models目录中

### 4.1. 测试训练模型的音频

音频输出到train/test_e2e，静态模型输出到train/inference

```shell
./synthesize_e2e.sh
```

### 4.2. fastspeech + pwg + multiple

```shell
python train/exps/synthesize_e2e.py \
        --am=fastspeech2_aishell3 \
        --am_config=train/conf/fastspeech2/default_multi.yaml \
        --am_ckpt=exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_<iter num>.pdz \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_aishell3 \
        --voc_config=pretrained_models/pwg_aishell3_ckpt_0.5/default.yaml \
        --voc_ckpt=pretrained_models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
        --voc_stat=pretrained_models/pwg_aishell3_ckpt_0.5/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --ngpu=1 \
        --spk_id=174
```

### 4.3. fastspeech + hifigan + single

```shell
python train/exps/synthesize_e2e.py \
        --am=fastspeech2_csmsc \
        --am_config=train/conf/fastspeech2/default_single.yaml \
        --am_ckpt=exp/fastspeech2_ghost/checkpoints/snapshot_iter_<iter num>.pdz \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --voc_config=pretrained_models/hifigan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=pretrained_models/hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz \
        --voc_stat=pretrained_models/hifigan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=dump/phone_id_map.txt \
        --ngpu=1
```

### 4.4. speedyspeech + pwg

```shell
python train/exps/synthesize_e2e.py \
        --am=speedyspeech_csmsc \
        --am_config=train/conf/speedyspeech/default.yaml \
        --am_ckpt=exp/speedyspeech_bili3_aishell3/checkpoints/snapshot_iter_<iter num>.pdz \
        --am_stat=dump/train/feats_stats.npy \
        --voc=pwgan_csmsc \
        --voc_config=pretrained_models/pwg_baker_ckpt_0.4/pwg_default.yaml \
        --voc_ckpt=pretrained_models/pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --voc_stat=pretrained_models/pwg_baker_ckpt_0.4/pwg_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt
```

## 5. GUI界面

![Alt text](gui/gui.png?raw=true "Title")

安装依赖库：

```shell
pip install PyQt5
pip install sounddevice
```

启动GUI界面：
（从项目根目录启动）

* 动态模型启动（开发者测试用）设置`self.use_static = False`
* 静态模型启动（用户使用）设置`self.use_static = True`

```shell
python gui/main.py
```

## 6. TODO list

- [x] 添加GST模块。
- [x] spleeter降噪。
- [ ] 静态模型推理。
- [ ] 优化ASR流程，目前batch size = 1，速度慢。
- [ ] preprocess优化，不需要重复处理数据集。
- [ ] 小规模数据集训练（AdaSpeech）。
- [ ] VAE情感控制。

## 7. FAQ

### 我啥也不懂，不是程序员，这个到底怎么用？

到时候发布模型的时候，我会一起出视频教程。

### 那你啥时候会发布模型？

尽量1月底吧，目前还在试模型效果，效果还没到最好。

### 我的电脑可以训练吗？

语音合成需要的显卡还是配置比较高的，我用的是3090的卡，我的建议是如果你要训练选择speedyspeech模型，这个模型比fastspeech2速度快很多，最终我发布的模型也会是speedyspeech。如果你是CPU的话那还是训练不起来的。

### 我训练好了，但是感觉效果很差？

可以从音源和vocoder的两个方向找问题，音源标注正确率低，有噪音，录音环境不一致，这些都是导致效果差的原因，而没有根据音源训练vocoder或者是没有finetune，都会导致效果很差。
