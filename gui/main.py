import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QComboBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# from paddle
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import jit
from paddle.static import InputSpec
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
    "style_melgan":
    "paddlespeech.t2s.models.melgan:StyleMelGANGenerator",
    "style_melgan_inference":
    "paddlespeech.t2s.models.melgan:StyleMelGANInference",
    "hifigan":
    "paddlespeech.t2s.models.hifigan:HiFiGANGenerator",
    "hifigan_inference":
    "paddlespeech.t2s.models.hifigan:HiFiGANInference",
}

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
        self.play_button = QPushButton('play', self)
        self.play_button.move(20, 120)
        self.play_button.clicked.connect(self.playAudioFile)

        # player
        self.player = QMediaPlayer()

        # voice combobox
        self.voice_label = QLabel(self)
        self.voice_label.move(160, 80)
        self.voice_label.setText("声音：")

        self.voice_combo = QComboBox(self)
        self.voice_combo.addItem("阿梓")
        self.voice_combo.addItem("老菊")
        self.voice_combo.addItem("海子姐")

        self.voice_combo.move(240, 80)
        self.voice_combo.resize(120, 40)
        self.voice_combo.activated[str].connect(self.onVoiceComboboxChanged)

        # tts model

        self.tts_model_label = QLabel(self)
        self.tts_model_label.move(160, 120)
        self.tts_model_label.setText("声学模型：")

        self.tts_model_combo = QComboBox(self)
        self.tts_model_combo.addItem("fastspeech2")
        # self.tts_model_combo.addItem("speedyspeech")

        self.tts_model_combo.move(240, 120)
        self.tts_model_combo.resize(120, 40)
        self.tts_model_combo.activated[str].connect(self.onTTSModelComboboxChanged)

        # vocoder model
        self.voc_model_label = QLabel(self)
        self.voc_model_label.move(160, 160)
        self.voc_model_label.setText("vocoder：")

        self.voc_model_combo = QComboBox(self)
        self.voc_model_combo.addItem("parallel wavegan")
        self.voc_model_combo.addItem("hifigan")

        self.voc_model_combo.move(240, 160)
        self.voc_model_combo.resize(120, 40)
        self.voc_model_combo.activated[str].connect(self.onVocModelComboboxChanged)  

        self.show()

    def initModel(self, tts_model=None):
        # settings
        
        self.lang = 'zh'

        if tts_model == None:
            self.am = "fastspeech2_aishell3" # self.am = "speedyspeech_csmsc"
        else:
            self.am = tts_model
        
        if (self.am == "fastspeech2_csmsc"):
            self.phones_dict = "../exp/fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt"
            self.tones_dict = None
            self.speaker_dict = "speaker_id_map.txt"
            self.am_ckpt = "../exp/fastspeech2_nosil_baker_ckpt_0.4/checkpoints/snapshot_iter_76000.pdz"
            self.am_stat = "../exp/fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy"
            with open("../exp/fastspeech2_nosil_baker_ckpt_0.4/default.yaml") as f:
                self.am_config = CfgNode(yaml.safe_load(f))
        elif (self.am == "speedyspeech_csmsc"):
            self.phones_dict = "../exp/speedyspeech_nosil_baker_ckpt_0.5/phone_id_map.txt"
            self.speaker_dict = "speaker_id_map.txt"
            self.am_ckpt = "../exp/speedyspeech_nosil_baker_ckpt_0.5/checkpoints/snapshot_iter_11400.pdz"
            self.am_stat = "../exp/speedyspeech_nosil_baker_ckpt_0.5/feats_stats.npy"
            with open("../exp/speedyspeech_nosil_baker_ckpt_0.5/default.yaml") as f:
                self.am_config = CfgNode(yaml.safe_load(f))
            self.tones_dict = "../exp/speedyspeech_nosil_baker_ckpt_0.5/tone_id_map.txt"
        elif (self.am == "fastspeech2_aishell3"):
            self.phones_dict = "../exp/fastspeech2_bili3_aishell3/phone_id_map.txt"
            self.tones_dict = None
            self.speaker_dict = "../exp/fastspeech2_bili3_aishell3/speaker_id_map.txt"
            self.am_ckpt = "../exp/fastspeech2_bili3_aishell3/checkpoints/snapshot_iter_13830.pdz"
            self.am_stat = "../exp/fastspeech2_bili3_aishell3/speech_stats.npy"
            with open("../exp/fastspeech2_bili3_aishell3/default_multi.yaml") as f:
                self.am_config = CfgNode(yaml.safe_load(f))
            self.spk_id = 174

        self.voc = "pwgan_csmsc"
        self.voc_config = "../pretrained_models/pwg_aishell3_ckpt_0.5/default.yaml"
        self.voc_ckpt = "../pretrained_models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz" 
        self.voc_stat = "../pretrained_models/pwg_aishell3_ckpt_0.5/feats_stats.npy"
        self.output_dir = "./"
        with open("../pwg_baker_ckpt_0.4/pwg_default.yaml") as f:
            self.voc_config = CfgNode(yaml.safe_load(f))

        # frontend
        if self.lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict, tone_vocab_path=self.tones_dict)
        elif self.lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        print("frontend done!")

        self.loadAcousticModel()

        self.loadVocoderModel()
    
    def loadAcousticModel(self):
        # acoustic model
        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        print("vocab_size:", vocab_size)

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)
            print("tone_size:", tone_size)

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            print("spk_num:", spk_num)

        odim = self.am_config.n_mels

        # model: {model_name}_{dataset}
        self.am_name = self.am[:self.am.rindex('_')]
        self.am_dataset = self.am[self.am.rindex('_') + 1:]

        am_class = dynamic_import(self.am_name, model_alias)
        am_inference_class = dynamic_import(self.am_name + '_inference', model_alias)

        if self.am_name == 'fastspeech2':
            am = am_class(
                idim=vocab_size, odim=odim, spk_num=spk_num, **self.am_config["model"])
        elif self.am_name == 'speedyspeech':
            am = am_class(
                vocab_size=vocab_size, tone_size=tone_size, **self.am_config["model"])

        am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        am.eval()
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        am_normalizer = ZScore(am_mu, am_std)
        self.am_inference = am_inference_class(am_normalizer, am)
        self.am_inference.eval()
        print("acoustic model done!")

    def loadVocoderModel(self):   
        # vocoder
        # model: {model_name}_{dataset}
        voc_name = self.voc[:self.voc.rindex('_')]
        voc_class = dynamic_import(voc_name, model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
        voc = voc_class(**self.voc_config["generator_params"])
        voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
        voc.remove_weight_norm()
        voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
        self.voc_inference.eval()
        print(voc_name + " voc done!")

    @pyqtSlot()
    def onGenerateButtonClicked(self):
        textboxValue = self.textbox.text()

        sentences = []
        sentences.append(("001", textboxValue))


        # whether dygraph to static
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for utt_id, sentence in sentences:
            if sentence == "":
                self.messageDialog('输入的文本不能为空，请检查。')
                continue
            try:
                get_tone_ids = False
                if self.am_name == 'speedyspeech':
                    get_tone_ids = True
                if self.lang == 'zh':
                    input_ids = self.frontend.get_input_ids(
                        sentence, merge_sentences=True, get_tone_ids=get_tone_ids)
                    phone_ids = input_ids["phone_ids"]
                    phone_ids = phone_ids[0]
                    if get_tone_ids:
                        tone_ids = input_ids["tone_ids"]
                        tone_ids = tone_ids[0]
                elif self.lang == 'en':
                    input_ids = self.frontend.get_input_ids(sentence)
                    phone_ids = input_ids["phone_ids"]
                else:
                    print("lang should in {'zh', 'en'}!")

                with paddle.no_grad():
                    # acoustic model
                    if self.am_name == 'fastspeech2':
                        # multi speaker
                        if self.am_dataset in {"aishell3", "vctk"}:
                            self.spk_id = paddle.to_tensor(self.spk_id)
                            mel = self.am_inference(phone_ids, self.spk_id)
                        else:
                            mel = self.am_inference(phone_ids)
                    elif self.am_name == 'speedyspeech':
                        mel = self.am_inference(phone_ids, tone_ids)
                    print("mel inference done.")
                    # vocoder
                    wav = self.voc_inference(mel)
                    print("vocoder inference done.")
                sf.write(
                    str(output_dir / ("output.wav")),
                    wav.numpy(),
                    samplerate=self.am_config.fs)
                print(f"write done.")
            except KeyError:
                print("输入的文本不能为空，请检查。")
                self.messageDialog('输入的文本不能为空，请检查。')
            except IndexError:
                print("输入的文本只支持中文，请检查。")
                self.messageDialog('输入的文本只支持中文，请检查。')
        self.playAudioFile()

    def onVoiceComboboxChanged(self, text):
        # self.qlabel.setText(text)
        # self.qlabel.adjustSize()
        if text == "阿梓":
            self.spk_id = 174
        elif text == "老菊":
            self.spk_id = 175
        elif text == "海子姐":
            self.spk_id = 176

    def onTTSModelComboboxChanged(self, text):
        if text == "fastspeech2":
            self.initModel("fastspeech2_csmsc")
        elif text == "speedyspeech":
            self.initModel("speedyspeech_csmsc")
        self.loadAcousticModel()

    def onVocModelComboboxChanged(self, text):
        if text == "parallel wavegan":
            self.voc = "pwgan_csmsc"
            self.voc_config = "../pwg_baker_ckpt_0.4/pwg_default.yaml"
            self.voc_ckpt = "../pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz"
            self.voc_stat = "../pwg_baker_ckpt_0.4/pwg_stats.npy"
            self.output_dir = "./"
            with open(self.voc_config) as f:
                self.voc_config = CfgNode(yaml.safe_load(f))
            self.loadVocoderModel()
        elif text == "hifigan":
            self.voc = "hifigan_csmsc"
            self.voc_config = "../hifigan_csmsc_ckpt_0.1.1/default.yaml"
            self.voc_ckpt = "../hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz"
            self.voc_stat = "../hifigan_csmsc_ckpt_0.1.1/feats_stats.npy"
            self.output_dir = "./"
            with open(self.voc_config) as f:
                self.voc_config = CfgNode(yaml.safe_load(f))
            self.loadVocoderModel()

    def playAudioFile(self):
        full_file_path = os.path.join(os.getcwd(), 'output.wav')
        url = QUrl.fromLocalFile(full_file_path)
        content = QMediaContent(url)
        print("play")
        print(content)
        self.player.setMedia(content)
        self.player.play()

    
    def messageDialog(self, text):
        msg_box = QMessageBox(QMessageBox.Warning, '错误', text)
        msg_box.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
