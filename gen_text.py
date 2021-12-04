# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from tqdm import tqdm

import paddle
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'gpu'])
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
# yapf: enable

def process(path, s2t_zh_model):
    if os.path.exists(path):
        files=os.listdir(path)
    else:
        print('this path not exist')
    for file in tqdm(files):
        if file.endswith('.wav'):
            print(file)
            file_name = os.path.splitext(file)[0]
            try:
                text = s2t_zh_model.speech_recognize(os.path.join(path, file))
                print(text)
                with open(os.path.join(path, file_name + ".txt"), "w") as f:
                    f.write(text)
            except ValueError as e:
                print("cannot recognize")
                os.replace(os.path.join(path, file), os.path.join(path, "unrecognized", file))

if __name__ == '__main__':
    paddle.set_device(args.device)

    # Check unrecognized exists or not
    isExist = os.path.exists(os.path.join(args.path, "unrecognized"))
    if not isExist:
        os.makedirs(os.path.join(args.path, "unrecognized"))
        print("unrecognized directory is created!")

    # s2t_zh_model = hub.Module(name='u2_conformer_aishell')
    s2t_zh_model = hub.Module(name='u2_conformer_librispeech')
    process(args.path, s2t_zh_model)

    # text_zh = s2t_zh_model.speech_recognize(args.wav_zh)
    # with open("test.txt","w") as f:
        # f.write(text_zh)