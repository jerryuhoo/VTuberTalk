import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QCheckBox, QLineEdit, QMessageBox, QComboBox, QLabel, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QUrl
import sounddevice as sd

# from paddle
import argparse
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

import sys

sys.path.append("train/frontend")
from zh_frontend import Frontend

sys.path.append("train/models")
from fastspeech2 import FastSpeech2
from speedyspeech import SpeedySpeech
# from paddlespeech.t2s.models.fastspeech2 import FastSpeech2

# from paddlespeech.t2s.models.fastspeech2 import StyleFastSpeech2Inference

from fastspeech2 import StyleFastSpeech2Inference
from speedyspeech import SpeedySpeechInference
from paddlespeech.t2s.models.hifigan import HiFiGANInference
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
import paddlespeech.t2s.models as ttsModels
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.data.get_feats import LogMelFBank

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

        self.tts_speed_combo = QComboBox(self)
        self.tts_speed_combo.addItem("1.0x")
        self.tts_speed_combo.addItem("0.8x")
        self.tts_speed_combo.addItem("1.2x")
        self.tts_speed_combo.addItem("古神")

        self.tts_speed_combo.move(240, 160)
        self.tts_speed_combo.resize(120, 40)
        self.tts_speed_combo.activated[str].connect(self.onTTSSpeedComboboxChanged)

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
        self.voc_model_combo.addItem("parallel wavegan")
        self.voc_model_combo.addItem("hifigan")

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
        self.speed = "1.0xspeed"
        self.wav = None

        if self.ngpu == 0:
            paddle.set_device("cpu")
        elif self.ngpu > 0:
            paddle.set_device("gpu")  

        self.voice_cloning = None

        self.onVoiceComboboxChanged(self.voice_combo.currentText())
        self.onTTSStyleComboboxChanged(self.tts_style_combo.currentText())
        self.onTTSSpeedComboboxChanged(self.tts_speed_combo.currentText())
        self.onAcousticModelComboboxChanged(self.acoustic_model_combo.currentText())
        self.onVocModelComboboxChanged(self.voc_model_combo.currentText())
        print("gst,", self.use_gst)
        print("vae,", self.use_vae)

    def loadFrontend(self):
        if self.acoustic_model == "fastspeech2":
            self.frontend = Frontend(phone_vocab_path=self.phones_dict)
        elif self.acoustic_model == "speedyspeech":
            self.frontend = Frontend(phone_vocab_path=self.phones_dict, tone_vocab_path=self.tones_dict)
        print("frontend done!")
    
    def loadAcousticModel(self):
        # acoustic model
        if self.acoustic_model == "fastspeech2":
            if self.use_gst:          
                self.fastspeech2_stat = "exp/gst_fastspeech2_azi_nanami/speech_stats.npy"
                self.fastspeech2_pitch_stat = "exp/gst_fastspeech2_azi_nanami/pitch_stats.npy"
                self.fastspeech2_energy_stat = "exp/gst_fastspeech2_azi_nanami/energy_stats.npy"
                self.phones_dict = "exp/gst_fastspeech2_azi_nanami/phone_id_map.txt"
                self.speaker_dict="exp/gst_fastspeech2_azi_nanami/speaker_id_map.txt"
                self.fastspeech2_config_path = "exp/gst_fastspeech2_azi_nanami/default_multi.yaml"
                self.fastspeech2_checkpoint = "exp/gst_fastspeech2_azi_nanami/checkpoints/snapshot_iter_111150.pdz"      
            elif self.use_vae:
                self.fastspeech2_stat = "exp/vae_fastspeech2_azi_nanami/speech_stats.npy"
                self.fastspeech2_pitch_stat = "exp/vae_fastspeech2_azi_nanami/pitch_stats.npy"
                self.fastspeech2_energy_stat = "exp/vae_fastspeech2_azi_nanami/energy_stats.npy"
                self.phones_dict = "exp/vae_fastspeech2_azi_nanami/phone_id_map.txt"
                self.speaker_dict="exp/vae_fastspeech2_azi_nanami/speaker_id_map.txt"
                self.fastspeech2_config_path = "exp/gst_fastspeech2_azi_nanami/default_multi.yaml"
                self.fastspeech2_checkpoint = "exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_165560.pdz"
            else:
                self.fastspeech2_stat = "exp/fastspeech2_bili3_aishell3/speech_stats.npy"
                self.fastspeech2_pitch_stat = "exp/fastspeech2_bili3_aishell3/pitch_stats.npy"
                self.fastspeech2_energy_stat = "exp/fastspeech2_bili3_aishell3/energy_stats.npy"
                self.phones_dict = "exp/fastspeech2_bili3_aishell3/phone_id_map.txt"
                self.speaker_dict="exp/fastspeech2_bili3_aishell3/speaker_id_map.txt"
                self.fastspeech2_config_path = "exp/gst_fastspeech2_azi_nanami/default_multi.yaml"
                self.fastspeech2_checkpoint = "exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_165560.pdz"

            with open(self.fastspeech2_config_path) as f:
                self.fastspeech2_config = CfgNode(yaml.safe_load(f))
        elif self.acoustic_model == "speedyspeech":
            self.speedyspeech_config_path = "exp/speedyspeech_azi_nanami_new/default_multi.yaml"
            self.speedyspeech_checkpoint = "exp/speedyspeech_azi_nanami_new/checkpoints/snapshot_iter_24037.pdz"
            self.speedyspeech_stat = "exp/speedyspeech_azi_nanami_new/feats_stats.npy"
            self.tones_dict = "exp/speedyspeech_azi_nanami_new/tone_id_map.txt"
            self.phones_dict = "exp/speedyspeech_azi_nanami_new/phone_id_map.txt"
            self.speaker_dict="exp/speedyspeech_azi_nanami_new/speaker_id_map.txt"
            with open(self.speedyspeech_config_path) as f:
                self.speedyspeech_config = CfgNode(yaml.safe_load(f))

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

        if self.acoustic_model == "fastspeech2":
            print("fastspeech2")
            odim = self.fastspeech2_config.n_mels
            self.model = FastSpeech2(
                idim=vocab_size, odim=odim, **self.fastspeech2_config["model"], spk_num=self.spk_num, use_gst=self.use_gst, use_vae=self.use_vae)

            self.model.set_state_dict(
                paddle.load(self.fastspeech2_checkpoint)["main_params"])

            self.model.eval()
            print("fastspeech2 model done!")
        elif self.acoustic_model == "speedyspeech":
            print("speedyspeech")
            tone_size = None
            if self.tones_dict:
                with open(self.tones_dict, "r") as f:
                    tone_id = [line.strip().split() for line in f.readlines()]
                tone_size = len(tone_id)
                print("tone_size:", tone_size)
            if self.use_gst:
                self.messageDialog("暂时不支持")
                return
            else:
                self.model = SpeedySpeech(
                    vocab_size=vocab_size, tone_size=tone_size, spk_num=self.spk_num, **self.speedyspeech_config["model"])

                self.model.set_state_dict(
                    paddle.load(self.speedyspeech_checkpoint)["main_params"])
            self.model.eval()
            print("speedyspeech model done!")
        

    def loadVocoderModel(self):   
        # vocoder
        class_map = {
            "hifigan": "HiFiGANGenerator",
            "mb_melgan": "MelGANGenerator",
            "pwgan": "PWGGenerator",
            "style_melgan": "StyleMelGANGenerator",
        }

        if self.vocoder == "pwg":
            self.pwg_config_path = "pretrained_models/pwg_aishell3_ckpt_0.5/default.yaml"
            self.pwg_checkpoint = "pretrained_models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz" 
            self.pwg_stat = "pretrained_models/pwg_aishell3_ckpt_0.5/feats_stats.npy"
            with open(self.pwg_config_path) as f:
                self.pwg_config = CfgNode(yaml.safe_load(f))
            checkpoint = self.pwg_checkpoint
            config = self.pwg_config
            generator_type = "pwgan"
        elif self.vocoder == "hifigan":
            self.hifigan_config_path = "pretrained_models/hifigan_aishell3/default.yaml"
            self.hifigan_checkpoint = "pretrained_models/hifigan_aishell3/snapshot_iter_390000.pdz" 
            self.hifigan_stat = "pretrained_models/hifigan_aishell3/feats_stats.npy"
            with open(self.hifigan_config_path) as f:
                self.hifigan_config = CfgNode(yaml.safe_load(f))
            checkpoint = self.hifigan_checkpoint
            config = self.hifigan_config
            generator_type = "hifigan"

        generator_class = getattr(ttsModels,
                              class_map[generator_type])
        self.generator = generator_class(**config["generator_params"])
        state_dict = paddle.load(checkpoint)
        self.generator.set_state_dict(state_dict["generator_params"])
        self.generator.remove_weight_norm()
        self.generator.eval()

        print("vocoder model done!")

    @pyqtSlot()
    def onGenerateButtonClicked(self):
        if self.ref_audio_path == "" and (self.use_gst or self.use_vae):
            self.messageDialog("请先选择参考音频！")
            return

        textboxValue = self.textbox.text()
        if textboxValue == "":
            self.messageDialog("输入不能为空！")
            return

        sentences = []
        sentences.append(("001", textboxValue))

        if self.acoustic_model == "fastspeech2":
            stat = np.load(self.fastspeech2_stat)
            mu, std = stat
            mu = paddle.to_tensor(mu)
            std = paddle.to_tensor(std)
            fastspeech2_normalizer = ZScore(mu, std)
        elif self.acoustic_model == "speedyspeech":
            stat = np.load(self.speedyspeech_stat)
            mu, std = stat
            mu = paddle.to_tensor(mu)
            std = paddle.to_tensor(std)
            speedyspeech_normalizer = ZScore(mu, std)

        if self.vocoder == "pwg":
            stat = np.load(self.pwg_stat)
        elif self.vocoder == "hifigan":
            stat = np.load(self.hifigan_stat)
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        vocoder_normalizer = ZScore(mu, std)

        if self.acoustic_model == "fastspeech2":
            fastspeech2_inference = StyleFastSpeech2Inference(
                fastspeech2_normalizer, self.model, self.fastspeech2_pitch_stat,
                self.fastspeech2_energy_stat)
            fastspeech2_inference.eval()
        elif self.acoustic_model == "speedyspeech":
            speedyspeech_inference = SpeedySpeechInference(
                speedyspeech_normalizer, self.model)
            speedyspeech_inference.eval()

        if self.vocoder == "pwg":
            vocoder_inference = PWGInference(vocoder_normalizer, self.generator)
        elif self.vocoder == "hifigan":
            vocoder_inference = HiFiGANInference(vocoder_normalizer, self.generator)
        vocoder_inference.eval()

        robot = False
        durations = None
        durations_scale = None
        durations_bias = None
        pitch = None
        pitch_scale = None
        pitch_bias = None
        energy = None
        energy_scale = None
        energy_bias = None

        if self.tts_style_combo.currentText() == "机器楞":
            self.style = "robot"
        elif self.tts_style_combo.currentText() == "高音":
            self.style = "high_voice"
        elif self.tts_style_combo.currentText() == "低音":
            self.style = "low_voice"

        if self.tts_speed_combo.currentText() == "1.2x":
            self.speed = "1.2xspeed"
        elif self.tts_speed_combo.currentText() == "0.8x":
            self.speed = "0.8xspeed"
        elif self.tts_speed_combo.currentText() == "古神":
            self.speed = "3.0xspeed"

        if self.style == "robot":
            # all tones in phones be `1`
            # all pitch should be the same, we use mean here
            robot = True
        if self.speed == "1.2xspeed":
            durations_scale = 1 / 1.2
        elif self.speed == "1.0xspeed":
            durations_scale = 1
        elif self.speed == "0.8xspeed":
            durations_scale = 1 / 0.8
        elif self.speed == "3.0xspeed":
            durations_scale = 1 / 3.0
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
                    mel = fastspeech2_inference(
                        phone_ids,
                        speech=speech,
                        durations=durations,
                        durations_scale=durations_scale,
                        durations_bias=durations_bias,
                        pitch=pitch,
                        pitch_scale=pitch_scale,
                        pitch_bias=pitch_bias,
                        energy=energy,
                        energy_scale=energy_scale,
                        energy_bias=energy_bias,
                        robot=robot,
                        spk_emb=None,
                        spk_id=self.spk_id
                        )
                elif self.acoustic_model == "speedyspeech":
                    tone_ids = paddle.to_tensor(input_ids["tone_ids"][0])
                    mel = speedyspeech_inference(
                        phone_ids,
                        tone_ids,
                        spk_id=self.spk_id
                        )
                print("mel infer done")
                self.wav = vocoder_inference(mel)
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
                sf.write(fpath, self.wav.numpy(), samplerate=self.fastspeech2_config.fs)
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
        if text == "阿梓":
            self.spk_id = 175
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
        
    def onTTSSpeedComboboxChanged(self, text):
        if text == "1.0x":
            self.speed = "1.0xspeed"
        elif text == "1.2x":
            self.speed = "1.2xspeed"
        elif text == "0.8x":
            self.speed = "0.8xspeed"
        elif text == "古神":
            self.speed = "3.0xspeed"

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
        if text == "parallel wavegan":
            self.vocoder = "pwg"
        elif text == "hifigan":
            self.vocoder = "hifigan"
        self.loadVocoderModel()

    def playAudioFile(self):
        if type(self.wav) == type(None):
            self.messageDialog("请先合成音频！")
            return
        try:
            sd.stop()
            sd.play(self.wav, self.fastspeech2_config.fs)
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