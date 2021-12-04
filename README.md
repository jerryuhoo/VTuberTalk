# VTuberTalk

## 1. 介绍

这是一个根据VTuber的声音训练而成的TTS（text-to-speech）模型，输入文本和VTuber可以输出对应的语音。本项目基于百度PaddleSpeech。

## 2. 预处理

### 2.1. 从直播录像中获取音频

从B站获取音频的方法：
可以用bilibili助手下载Vtuber的录播flv文件，再转成wav文件。

从YouTube获取音频的方法：
可以用TamperMonkey上的YouTube下载器下载mp4文件，再转成wav文件。

```
python tools/video_to_mp3.py --path data
```

其中，在video_to_mp3可设置采样率，一般设置为16000，因为如果要使用语音切分工具的话，16000是支持的采样率之一。

### 2.2. 将音频分割成片段

步骤2.2和2.3仅限于没有字幕的音频，如果在YouTube下载的话大概率会有字幕文件，下载字幕文件后直接跳转到“2.4. 使用字幕获得文本”即可。

音频分割使用了webrtcvad模块，其中第一个参数aggressiveness是分割检测的敏感度，数字越大，对于静音检测越敏感，分割的音频个数也越多。范围为0～3。

```
python tools/split_audio.py <aggressiveness>(0~3) <path to wav file>
```

### 2.3. 使用ASR获得文本

目前只支持英文的识别，中文识别尚不准确。

```
python gen_text.py --path data
```

### 2.4. 使用字幕获得文本

## 3. 训练

## 4. 推理
