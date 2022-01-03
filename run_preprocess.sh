set -e

speaker=azi
cut_minute=10
sample_rate=32000
ag=3
min_text=2
max_text=60
use_spleeter=False

stage=5
stop_stage=6

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # cut video
    echo $stage
    echo "cut video"
    python tools/cut_source.py --path data/wav_temp/$speaker/video/ --min $cut_minute --sr $sample_rate || exit -1
    cd data/wav_temp/$speaker/raw
    find . | grep 'wav' | nl -nrz -w3 -v1 | while read n f; do mv "$f" "${speaker}_$n.wav"; done || exit -1
    cd ../../../../
fi

if [ ${use_spleeter} == True ]; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        # spleeter
        echo "spleeter"
        spleeter separate -o data/wav_temp/$speaker/clean_raw data/wav_temp/$speaker/raw/*.wav || exit -1
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # glob spleeter vocals
        echo "glob spleeter vocals"
        python tools/glob_spleeter_vocals.py --path data/wav_temp/$speaker/clean_raw/ || exit -1
        rm -rf data/wav_temp/$speaker/clean_raw/ || exit -1
        python tools/audio_to_mono.py --path data/wav_temp/$speaker/clean_raw2/ || exit -1
        mv data/wav_temp/$speaker/clean_raw2/ data/wav_temp/$speaker/raw/ || exit -1
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # split
    echo "split"
    python tools/split_audio.py --ag $ag --in_path data/wav_temp/$speaker/raw/ || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # asr
    cp -r data/wav_temp/$speaker/split/ data/wav_temp/$speaker/split16000/ || exit -1
    echo "change sample rate to 16000 for asr"
    python tools/change_sr.py --path data/wav_temp/$speaker/split16000/ --sr 16000 || exit -1
    echo "asr"
    python tools/gen_text.py --path data/wav_temp/$speaker/split16000/ || exit -1
    mv data/wav_temp/$speaker/split16000/*.txt data/wav_temp/$speaker/split/ || exit -1
    rm -rf data/wav_temp/$speaker/split16000/ || exit -1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # clean
    echo "glob text"
    python tools/glob_text.py --path data/wav_temp/$speaker/split/ || exit -1
    echo "data filter"
    python tools/data_filter.py --path data/wav_temp/$speaker/split/ --min $min_text --max $max_text || exit -1
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # hanzi to pinyin
    echo "hanzi to pinyin"
    python tools/hanzi_to_pinyin.py --path data/wav_temp/$speaker/split/ || exit -1
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "mfa training"
    mfa train data/wav_temp/$speaker/split/ MFA/pinyin_eng.dict MFA/$speaker.zip data/TextGrid_temp/$speaker/ --clean || exit -1
fi
