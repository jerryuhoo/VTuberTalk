import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QCheckBox, QLineEdit, QMessageBox, QComboBox, QLabel, QFileDialog
from PyQt5.QtCore import pyqtSlot
import sounddevice as sd

# from paddle
import argparse
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf

import sys

sys.path.append("train/frontend")
from zh_frontend import Frontend

sys.path.append("train/models")
# from paddlespeech.t2s.models.fastspeech2 import FastSpeech2

# from paddlespeech.t2s.models.fastspeech2 import StyleFastSpeech2Inference

from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.datasets.get_feats import LogMelFBank

import librosa
from sklearn.preprocessing import StandardScaler

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'VTuberTalk'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 320
        self.initUI()
        self.initModel()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(360, 40)
        
        # generate button
        self.generate_button = QPushButton('合成', self)
        self.generate_button.move(20, 80)
        self.generate_button.clicked.connect(self.onGenerateButtonClicked)
        
        # play button
        self.play_button = QPushButton('重播', self)
        self.play_button.move(20, 120)
        self.play_button.clicked.connect(self.playAudioFile)

        # save button
        self.save_button = QPushButton('保存', self)
        self.save_button.move(20, 160)
        self.save_button.clicked.connect(self.saveWavFile)

        # voice combobox
        self.voice_label = QLabel(self)
        self.voice_label.move(160, 80)
        self.voice_label.setText("声音：")

        self.voice_combo = QComboBox(self)
        self.voice_combo.addItem("阿梓")
        self.voice_combo.addItem("海子姐")

        self.voice_combo.move(240, 80)
        self.voice_combo.resize(120, 40)
        self.voice_combo.activated[str].connect(self.onVoiceComboboxChanged)

        # tts model

        self.tts_style_label = QLabel(self)
        self.tts_style_label.move(160, 120)
        self.tts_style_label.setText("风格：")

        self.tts_style_combo = QComboBox(self)
        self.tts_style_combo.addItem("正常")
        self.tts_style_combo.addItem("机器楞")
        self.tts_style_combo.addItem("高音")
        self.tts_style_combo.addItem("低音")

        self.tts_style_combo.move(240, 120)
        self.tts_style_combo.resize(120, 40)
        self.tts_style_combo.activated[str].connect(self.onTTSStyleComboboxChanged)

        self.tts_speed_label = QLabel(self)
        self.tts_speed_label.move(160, 160)
        self.tts_speed_label.setText("速度：")

        self.tts_speed_box = QLineEdit(self)
        self.tts_speed_box.setText("1.0")
        self.tts_speed_box.move(240, 160)
        self.tts_speed_box.resize(120, 40)

        #  acoustic model
        self.acoustic_model_label = QLabel(self)
        self.acoustic_model_label.move(160, 200)
        self.acoustic_model_label.setText("模型：")

        self.acoustic_model_combo = QComboBox(self)
        self.acoustic_model_combo.addItem("gst-fastspeech2")
        self.acoustic_model_combo.addItem("fastspeech2")
        self.acoustic_model_combo.addItem("gst-speedyspeech")
        self.acoustic_model_combo.addItem("speedyspeech")
        self.acoustic_model_combo.addItem("vae-fastspeech2")

        self.acoustic_model_combo.move(240, 200)
        self.acoustic_model_combo.resize(120, 40)
        self.acoustic_model_combo.activated[str].connect(self.onAcousticModelComboboxChanged)

        # # model path
        # self.ref_audio_button = QPushButton('加载模型路径', self)
        # self.ref_audio_button.move(20, 200)
        # self.ref_audio_button.clicked.connect(self.loadRefWavFile)

        # vocoder model
        self.voc_model_label = QLabel(self)
        self.voc_model_label.move(160, 240)
        self.voc_model_label.setText("vocoder：")

        self.voc_model_combo = QComboBox(self)
        self.voc_model_combo.addItem("hifigan_fs")
        self.voc_model_combo.addItem("hifigan_ss")

        self.voc_model_combo.move(240, 240)
        self.voc_model_combo.resize(120, 40)
        self.voc_model_combo.activated[str].connect(self.onVocModelComboboxChanged)

        # ref audio
        self.ref_audio_button = QPushButton('参考音频', self)
        self.ref_audio_button.move(20, 240)
        self.ref_audio_button.clicked.connect(self.loadRefWavFile)

        self.ref_audio_label = QLabel(self)
        self.ref_audio_label.move(160, 280)
        self.ref_audio_label.resize(380, 40)
        self.ref_audio_label.setText("未加载参考音频")
        self.ref_audio_path = ""

        self.show()

    def initModel(self, tts_model=None):
        # settings
        # parse args and config and redirect to train_sp
        self.ngpu = 0
        self.style = "Normal"
        self.speed = 1.0
        self.wav = None
        self.fs = 24000

        if self.ngpu == 0:
            paddle.set_device("cpu")
        elif self.ngpu > 0:
            paddle.set_device("gpu")  

        self.voice_cloning = None

        self.onTTSStyleComboboxChanged(self.tts_style_combo.currentText())
        self.onAcousticModelComboboxChanged(self.acoustic_model_combo.currentText())
        self.onVocModelComboboxChanged(self.voc_model_combo.currentText())
        self.onVoiceComboboxChanged(self.voice_combo.currentText())
        print("gst,", self.use_gst)
        print("vae,", self.use_vae)

    def loadFrontend(self):
        if self.acoustic_model == "fastspeech2":
            try:
                self.frontend = Frontend(phone_vocab_path=self.phones_dict)
            except:
                self.messageDialog("未找到phones_dict路径")
        elif self.acoustic_model == "speedyspeech":
            try:
                self.frontend = Frontend(phone_vocab_path=self.phones_dict, tone_vocab_path=self.tones_dict)
            except:
                self.messageDialog("未找到phones_dict路径")
        print("frontend done!")
    
    def loadAcousticModel(self):
        # acoustic model
        if self.acoustic_model == "fastspeech2":
            self.phones_dict = "pretrained_models/style_fastspeech2_azi_nanami_static/phone_id_map.txt"
            self.speaker_dict = "pretrained_models/style_fastspeech2_azi_nanami_static/speaker_id_map.txt"
            self.am_inference = paddle.jit.load(
                            os.path.join("pretrained_models/style_fastspeech2_azi_nanami_static", "fastspeech2_aishell3"))
                
        elif self.acoustic_model == "speedyspeech":
            self.tones_dict = "pretrained_models/speedyspeech_azi_nanami_static/tone_id_map.txt"
            self.phones_dict = "pretrained_models/speedyspeech_azi_nanami_static/phone_id_map.txt"
            self.speaker_dict="pretrained_models/speedyspeech_azi_nanami_static/speaker_id_map.txt"
            self.am_inference = paddle.jit.load(
                            os.path.join("pretrained_models/speedyspeech_azi_nanami_static", "speedyspeech_csmsc"))

        fields = ["utt_id", "text"]
        self.spk_num = None
        if self.speaker_dict:
            print("multiple speaker")
            with open(self.speaker_dict, 'rt') as f:
                spk_id_list = [line.strip().split() for line in f.readlines()]
            self.spk_num = len(spk_id_list)
            fields += ["spk_id"]
        elif self.voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
        else:
            print("single speaker")
        print("spk_num:", self.spk_num)

        with open(self.phones_dict, "r", encoding='UTF-8') as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        print("vocab_size:", vocab_size)


    def loadVocoderModel(self):   
        # vocoder
        if self.vocoder == "hifigan_fs":
            self.voc_inference = paddle.jit.load(os.path.join("pretrained_models/style_fastspeech2_azi_nanami_static", "hifigan_csmsc"))
        elif self.vocoder == "hifigan_ss":
            self.voc_inference = paddle.jit.load(os.path.join("pretrained_models/speedyspeech_azi_nanami_static", "hifigan_csmsc"))

    @pyqtSlot()
    def onGenerateButtonClicked(self):
        if self.ref_audio_path == "" and (self.use_gst or self.use_vae):
            self.messageDialog("请先选择参考音频！")
            return

        textboxValue = self.textbox.text()
        if textboxValue == "":
            self.messageDialog("文字输入不能为空！")
            return

        speed_value = self.tts_speed_box.text()
        if speed_value == "":
            self.messageDialog("速度输入不能为空！")
            return
        else:
            try:
                self.speed = min(3.0, max(0.1, float(speed_value)))
            except ValueError:
                self.messageDialog("输入错误，需要输入整数或小数，正常速度为1.0！")

        sentences = []
        sentences.append(("001", textboxValue))

        robot = False
        durations = None
        durations_scale = 1.0
        durations_bias = 0.0
        pitch = None
        pitch_scale = 1.0
        pitch_bias = 0.0
        energy = None
        energy_scale = 1.0
        energy_bias = 0.0

        if self.tts_style_combo.currentText() == "机器楞":
            self.style = "robot"
        elif self.tts_style_combo.currentText() == "高音":
            self.style = "high_voice"
        elif self.tts_style_combo.currentText() == "低音":
            self.style = "low_voice"

        if self.style == "robot":
            # all tones in phones be `1`
            # all pitch should be the same, we use mean here
            robot = True
        
        durations_scale = self.speed

        if self.style == "high_voice":
            pitch_scale = 1.3
        elif self.style == "low_voice":
            pitch_scale = 0.7
        elif self.style == "normal":
            pitch_scale = 1
         
        record = None
        try:
            wav, _ = librosa.load(str(self.ref_audio_path), sr=self.fastspeech2_config.fs)
            if len(wav.shape) != 1 or np.abs(wav).max() > 1.0:
                return record
            assert len(wav.shape) == 1, f"ref audio is not a mono-channel audio."
            assert np.abs(wav).max(
            ) <= 1.0, f"ref audio is seems to be different that 16 bit PCM."

            mel_extractor = LogMelFBank(
                sr=self.fastspeech2_config.fs,
                n_fft=self.fastspeech2_config.n_fft,
                hop_length=self.fastspeech2_config.n_shift,
                win_length=self.fastspeech2_config.win_length,
                window=self.fastspeech2_config.window,
                n_mels=self.fastspeech2_config.n_mels,
                fmin=self.fastspeech2_config.fmin,
                fmax=self.fastspeech2_config.fmax)

            logmel = mel_extractor.get_log_mel_fbank(wav)
            # normalize, restore scaler
            speech_scaler = StandardScaler()
            speech_scaler.mean_ = np.load(self.fastspeech2_stat)[0]
            speech_scaler.scale_ = np.load(self.fastspeech2_stat)[1]
            speech_scaler.n_features_in_ = speech_scaler.mean_.shape[0]
            logmel = speech_scaler.transform(logmel)
            
            speech = paddle.to_tensor(logmel)
        except:
            speech = None
        
        for utt_id, sentence in sentences:
            if self.acoustic_model == "fastspeech2":
                input_ids = self.frontend.get_input_ids(
                    sentence, merge_sentences=True, robot=robot)
            elif self.acoustic_model == "speedyspeech":
                input_ids = self.frontend.get_input_ids(
                sentence, merge_sentences=True, get_tone_ids=True)
            try:
                phone_ids = input_ids["phone_ids"][0]
            except:
                self.messageDialog("输入的文字不能识别，请重新输入！")
                return

            print("self.spk_id", self.spk_id)

            self.spk_id = paddle.to_tensor(self.spk_id)

            # self.spk_id = None # temp

            with paddle.no_grad():
                if self.acoustic_model == "fastspeech2":
                    d_scale = paddle.to_tensor(durations_scale, dtype='float32')
                    p_scale = paddle.to_tensor(pitch_scale, dtype='float32')
                    e_scale = paddle.to_tensor(energy_scale, dtype='float32')
                    d_bias = paddle.to_tensor(durations_bias, dtype='float32')
                    p_bias = paddle.to_tensor(pitch_bias, dtype='float32')
                    e_bias = paddle.to_tensor(energy_bias, dtype='float32')
                    robot = paddle.to_tensor(robot, dtype='bool')
                    mel = self.am_inference(phone_ids, d_scale, d_bias, p_scale, p_bias, e_scale, e_bias, robot, self.spk_id)
                elif self.acoustic_model == "speedyspeech":
                    tone_ids = paddle.to_tensor(input_ids["tone_ids"][0])
                    mel = self.am_inference(phone_ids, tone_ids, self.spk_id)
                    print("mel", mel)
                    print("mel shape", mel.shape)
                print("mel infer done")
                self.wav = self.voc_inference(mel)
                print("vocoder infer done")
            print(f"{self.style}_{utt_id} done!")

        self.playAudioFile()

    def saveWavFile(self):
        if type(self.wav) != type(None):
            dialog = QFileDialog()
            dialog.setDefaultSuffix(".wav")
            fpath, _ = dialog.getSaveFileName(
                parent=self,
                caption="Select a path to save the audio file",
                filter="Audio Files (*.flac *.wav)"
            )
            if fpath:
                if Path(fpath).suffix == "":
                    fpath += ".wav"
                sf.write(fpath, self.wav.numpy(), samplerate=self.fs)
        else:
            self.messageDialog("还没有合成声音，无法保存！")
    
    def loadRefWavFile(self):
        '''
        setFileMode():
        QFileDialog.AnyFile,任何文件
        QFileDialog.ExistingFile,已存在的文件
        QFileDialog.Directory,文件目录
        QFileDialog.ExistingFiles,已经存在的多个文件
        '''
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        # dlg.setFilter(QDir.Files)

        if dialog.exec_():
            filenames= dialog.selectedFiles()
            self.ref_audio_path = filenames[0]
            self.ref_audio_label.setText("已加载：" + os.path.basename(filenames[0]))


    def onVoiceComboboxChanged(self, text):
        if self.acoustic_model == "speedyspeech":
            if text == "阿梓":
                self.spk_id = 175
            elif text == "海子姐":
                self.spk_id = 176
        elif self.acoustic_model == "fastspeech2":
            if text == "阿梓":
                self.spk_id = 174
            elif text == "海子姐":
                self.spk_id = 176

    def onTTSStyleComboboxChanged(self, text):
        if text == "正常":
            self.style = "normal"
        elif text == "机器楞":
            self.style = "robot"
        elif text == "高音":
            self.style = "high_voice"
        elif text == "低音":
            self.style = "low_voice"

    def onAcousticModelComboboxChanged(self, text):
        if text == "gst-fastspeech2":
            self.acoustic_model = "fastspeech2"
            self.use_gst = True
            self.use_vae = False
        elif text == "fastspeech2":
            self.acoustic_model = "fastspeech2"
            self.use_gst = False
            self.use_vae = False
        elif text == "gst-speedyspeech":
            self.messageDialog("暂不支持")
            return
        elif text == "speedyspeech":
            self.acoustic_model = "speedyspeech"
            self.use_gst = False
        elif text == "vae-fastspeech2":
            self.acoustic_model = "fastspeech2"
            self.use_vae = True
            self.use_gst = False
        self.onVoiceComboboxChanged(self.voice_combo.currentText())
        self.loadAcousticModel()
        self.loadFrontend()

    def onVocModelComboboxChanged(self, text):
        if text == "hifigan_fs":
            self.vocoder = "hifigan_fs"
        elif text == "hifigan_ss":
            self.vocoder = "hifigan_ss"
        self.loadVocoderModel()

    def playAudioFile(self):
        if type(self.wav) == type(None):
            self.messageDialog("请先合成音频！")
            return
        try:
            sd.stop()
            sd.play(self.wav, self.fs)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")

    def messageDialog(self, text):
        msg_box = QMessageBox(QMessageBox.Warning, '错误', text)
        msg_box.exec_()

    # def onClickedGST(self):
    #     if self.use_gst_button.isChecked():
    #         self.use_gst = True
    #     else:
    #         self.use_gst = False
    #     self.loadAcousticModel()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())