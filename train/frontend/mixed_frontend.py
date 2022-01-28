import re

# from paddlespeech.t2s.frontend.zh_frontend import Frontend as ChineseFrontend
# from paddlespeech.t2s.frontend.phonectic import English as EnglishFrontend
import sys
sys.path.append("train")
from frontend.zh_frontend import Frontend as ChineseFrontend
from frontend.phonectic import English as EnglishFrontend


class Frontend():
    def __init__(self,
                 g2p_model="pypinyin",
                 phone_vocab_path=None,
                 tone_vocab_path=None):
        print(phone_vocab_path)
        self.english_frontend = EnglishFrontend(phone_vocab_path=phone_vocab_path)
        self.chinese_frontend = ChineseFrontend(phone_vocab_path=phone_vocab_path, tone_vocab_path=tone_vocab_path)
    
    def spliteKeyWord(self, str):
        regex = r"[\u4e00-\ufaff|0-9|\W]+|[a-zA-Z\s]+"
        matches = re.findall(regex, str, re.UNICODE)
        return matches

    def isChineseOrNumber(self, str):
        return bool(re.search('[\u4e00-\ufaff|0-9|\W]+', str))

    def isEnglish(self, str):
        return bool(re.search('[a-zA-Z]+', str))

    def get_input_ids(self, sentence, merge_sentences, robot, get_tone_ids=False):
        segments = self.spliteKeyWord(sentence)
        result = {}
        for segment in segments:
            if(self.isEnglish(segment)):
                eng_text = self.english_frontend.get_input_ids(
                        segment, merge_sentences=merge_sentences)
                if result:
                    x1 = result["phone_ids"][0]
                    x2 = eng_text["phone_ids"][0]
                    result["phone_ids"][0] = paddle.concat(x=[x1, x2], axis=0)
                    if get_tone_ids:
                        x1 = result["tone_ids"][0]
                        x2 = paddle.to_tensor(0)
                        result["tone_ids"][0] = paddle.concat(x=[x1, x2], axis=0)
                else:
                    result["phone_ids"] = eng_text["phone_ids"]
                    if get_tone_ids:
                        result["tone_ids"] = paddle.to_tensor(0)
            elif(self.isChineseOrNumber(segment)):
                cn_text = self.chinese_frontend.get_input_ids(
                    segment, merge_sentences=merge_sentences, get_tone_ids=get_tone_ids, robot=robot)
                if result:
                    x1 = result["phone_ids"][0]
                    x2 = cn_text["phone_ids"][0]
                    result["phone_ids"][0] = paddle.concat(x=[x1, x2], axis=0)
                    if get_tone_ids:
                        x1 = result["tone_ids"][0]
                        x2 = paddle.to_tensor(0)
                        result["tone_ids"][0] = paddle.concat(x=[x1, x2], axis=0)
                else:
                    result["phone_ids"] = cn_text["phone_ids"]
                    if get_tone_ids:
                        result["tone_ids"] = cn_text["tone_ids"]
        return result