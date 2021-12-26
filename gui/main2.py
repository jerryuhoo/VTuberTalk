import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QComboBox, QLabel, QFileDialog
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

from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
# from paddlespeech.t2s.models.fastspeech2 import StyleFastSpeech2Inference
sys.path.append("..") 
from train.models.fastspeech2 import StyleFastSpeech2Inference
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.modules.normalizer import ZScore


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'VTuberTalk'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 300
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
        self.generate_button = QPushButton('generate', self)
        self.generate_button.move(20, 80)
        self.generate_button.clicked.connect(self.onGenerateButtonClicked)
        
        # play button
        self.play_button = QPushButton('replay', self)
        self.play_button.move(20, 120)
        self.play_button.clicked.connect(self.playAudioFile)

        # save button
        self.save_button = QPushButton('save', self)
        self.save_button.move(20, 160)
        self.save_button.clicked.connect(self.saveWavFile)

        # voice combobox
        self.voice_label = QLabel(self)
        self.voice_label.move(160, 80)
        self.voice_label.setText("声音：")

        self.voice_combo = QComboBox(self)
        self.voice_combo.addItem("阿梓")
        self.voice_combo.addItem("海子姐")
        self.voice_combo.addItem("老菊")

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

        # vocoder model
        self.voc_model_label = QLabel(self)
        self.voc_model_label.move(160, 200)
        self.voc_model_label.setText("vocoder：")

        self.voc_model_combo = QComboBox(self)
        self.voc_model_combo.addItem("parallel wavegan")
        self.voc_model_combo.addItem("hifigan")

        self.voc_model_combo.move(240, 200)
        self.voc_model_combo.resize(120, 40)
        self.voc_model_combo.activated[str].connect(self.onVocModelComboboxChanged)  

        self.show()

    def initModel(self, tts_model=None):
        # settings
        # parse args and config and redirect to train_sp
        
        self.fastspeech2_config_path = "../exp/fastspeech2_bili3_aishell3/default_multi.yaml"
        self.fastspeech2_checkpoint = "../exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_179790.pdz"
        self.fastspeech2_stat = "../exp/fastspeech2_bili3_aishell3/speech_stats.npy"
        self.fastspeech2_pitch_stat = "../exp/fastspeech2_bili3_aishell3/pitch_stats.npy"
        self.fastspeech2_energy_stat = "../exp/fastspeech2_bili3_aishell3/energy_stats.npy"
        self.pwg_config_path = "../pretrained_models/pwg_aishell3_ckpt_0.5/default.yaml"
        self.pwg_checkpoint = "../pretrained_models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz" 
        self.pwg_stat = "../pretrained_models/pwg_aishell3_ckpt_0.5/feats_stats.npy"
        self.phones_dict = "../exp/fastspeech2_bili3_aishell3/phone_id_map.txt"
        self.ngpu = 0
        self.style = "Normal"
        self.speed = "1.0xspeed"
        self.speaker_dict="../exp/fastspeech2_bili3_aishell3/speaker_id_map.txt"
        self.spk_id = 174
        self.wav = None


        if self.ngpu == 0:
            paddle.set_device("cpu")
        elif self.ngpu > 0:
            paddle.set_device("gpu")

        with open(self.fastspeech2_config_path) as f:
            self.fastspeech2_config = CfgNode(yaml.safe_load(f))
        with open(self.pwg_config_path) as f:
            self.pwg_config = CfgNode(yaml.safe_load(f))

        self.voice_cloning = None

        fields = ["utt_id", "text"]
        self.spk_num = None
        if self.speaker_dict:
            print("multiple speaker fastspeech2!")
            with open(self.speaker_dict, 'rt') as f:
                spk_id_list = [line.strip().split() for line in f.readlines()]
            self.spk_num = len(spk_id_list)
            fields += ["spk_id"]
        elif self.voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
        else:
            print("single speaker fastspeech2!")
        print("spk_num:", self.spk_num)

        self.loadAcousticModel()
        self.loadVocoderModel()

        self.frontend = Frontend(phone_vocab_path=self.phones_dict)
        print("frontend done!") 
    
    def loadAcousticModel(self):
        # acoustic model
        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        print("vocab_size:", vocab_size)

        odim = self.fastspeech2_config.n_mels
        self.model = FastSpeech2(
            idim=vocab_size, odim=odim, **self.fastspeech2_config["model"], spk_num=self.spk_num)

        self.model.set_state_dict(
            paddle.load(self.fastspeech2_checkpoint)["main_params"])
        self.model.eval()
        print("fastspeech2 model done!")

    def loadVocoderModel(self):   
        # vocoder
        self.vocoder = PWGGenerator(**self.pwg_config["generator_params"])
        self.vocoder.set_state_dict(paddle.load(self.pwg_checkpoint)["generator_params"])
        self.vocoder.remove_weight_norm()
        self.vocoder.eval()
        print("vocoder model done!")

    @pyqtSlot()
    def onGenerateButtonClicked(self):
        textboxValue = self.textbox.text()

        sentences = []
        sentences.append(("001", textboxValue))

        stat = np.load(self.fastspeech2_stat)
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        fastspeech2_normalizer = ZScore(mu, std)

        stat = np.load(self.pwg_stat)
        mu, std = stat
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        pwg_normalizer = ZScore(mu, std)

        fastspeech2_inference = StyleFastSpeech2Inference(
            fastspeech2_normalizer, self.model, self.fastspeech2_pitch_stat,
            self.fastspeech2_energy_stat)
        fastspeech2_inference.eval()

        pwg_inference = PWGInference(pwg_normalizer, self.vocoder)
        pwg_inference.eval()

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

        if self.tts_style_combo.currentText == "机器楞":
            self.style = "robot"
        elif self.tts_style_combo.currentText == "高音":
            self.style = "high_voice"
        elif self.tts_style_combo.currentText == "低音":
            self.style = "low_voice"

        if self.tts_speed_combo.currentText == "1.2x":
            self.speed = "1.2xspeed"
        elif self.tts_speed_combo.currentText == "0.8x":
            self.speed = "0.8xspeed"
        elif self.tts_speed_combo.currentText == "古神":
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
        
        for utt_id, sentence in sentences:
            input_ids = self.frontend.get_input_ids(
                sentence, merge_sentences=True, robot=robot)
            phone_ids = input_ids["phone_ids"][0]
            print("self.spk_id", self.spk_id)
            with paddle.no_grad():
                mel = fastspeech2_inference(
                    phone_ids,
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
                self.wav = pwg_inference(mel)
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

    def onVoiceComboboxChanged(self, text):
        if text == "阿梓":
            self.spk_id = 174
        elif text == "老菊":
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

    def onVocModelComboboxChanged(self, text):
        if text == "parallel wavegan":
            pass
        elif text == "hifigan":
            pass

    def playAudioFile(self):
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())