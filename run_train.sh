set -e

stage=0
stop_stage=100
model_name=fastspeech2_aishell3_english
fastspeech2=True
multiple=True
use_gst=False
use_vae=False

# config
config=default_multi # default_multi # conformer

if [ ${fastspeech2} == True ] && [ ${multiple} == False ]; then
    echo "model: fastspeech2, single"

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        echo "duration"
        python tools/gen_duration_from_textgrid.py \
            --inputdir=data/TextGrid \
            --output=data/durations.txt \
            --config=train/conf/fastspeech2/default_single.yaml || exit -1
    fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "preprocess"
        python train/exps/fastspeech2/preprocess.py \
            --dataset=other \
            --rootdir=data/ \
            --dumpdir=dump \
            --dur-file=data/durations.txt \
            --config=train/conf/fastspeech2/default_single.yaml \
            --num-cpu=16 \
            --cut-sil=True || exit -1
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "compute statistics"
        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="speech" || exit -1

        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="pitch" || exit -1

        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="energy" || exit -1
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo "normalize"
        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --dumpdir=dump/train/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt || exit -1

        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/dev/raw/metadata.jsonl \
            --dumpdir=dump/dev/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt || exit -1

        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/test/raw/metadata.jsonl \
            --dumpdir=dump/test/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt || exit -1
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4]; then
        echo "train"
        python train/exps/fastspeech2/train.py \
            --train-metadata=dump/train/norm/metadata.jsonl \
            --dev-metadata=dump/dev/norm/metadata.jsonl \
            --config=train/conf/fastspeech2/default_single.yaml \
            --output-dir=exp/$model_name \
            --ngpu=1 \
            --phones-dict=dump/phone_id_map.txt \
            --use_gst=$use_gst \
            --use_vae=$use_vae || exit -1
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "copy files to model"
        cp dump/phone_id_map.txt exp/$model_name/phone_id_map.txt
        cp dump/speaker_id_map.txt exp/$model_name/speaker_id_map.txt
        cp dump/train/energy_stats.npy exp/$model_name/energy_stats.npy
        cp dump/train/pitch_stats.npy exp/$model_name/pitch_stats.npy
        cp dump/train/speech_stats.npy exp/$model_name/speech_stats.npy
    fi
fi


if [ ${fastspeech2} == True ] && [ ${multiple} == True ]; then
    echo "model: fastspeech2, multiple"

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        echo "duration"
        python tools/gen_duration_from_textgrid.py \
            --inputdir=data/TextGrid \
            --output=data/durations.txt \
            --config=train/conf/fastspeech2/$config.yaml || exit -1
    fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "preprocess"
        python train/exps/fastspeech2/preprocess.py \
            --dataset=other \
            --rootdir=data/ \
            --dumpdir=dump \
            --dur-file=data/durations.txt \
            --config=train/conf/fastspeech2/$config.yaml \
            --num-cpu=16 \
            --cut-sil=True || exit -1
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "compute statistics"
        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="speech" || exit -1

        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="pitch" || exit -1

        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="energy" || exit -1
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo "normalize"
        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --dumpdir=dump/train/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt || exit -1

        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/dev/raw/metadata.jsonl \
            --dumpdir=dump/dev/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt || exit -1

        python train/exps/fastspeech2/normalize.py \
            --metadata=dump/test/raw/metadata.jsonl \
            --dumpdir=dump/test/norm \
            --speech-stats=dump/train/speech_stats.npy \
            --pitch-stats=dump/train/pitch_stats.npy \
            --energy-stats=dump/train/energy_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt || exit -1
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        echo "train"
        python train/exps/fastspeech2/train.py \
            --train-metadata=dump/train/norm/metadata.jsonl \
            --dev-metadata=dump/dev/norm/metadata.jsonl \
            --config=train/conf/fastspeech2/$config.yaml \
            --output-dir=exp/$model_name \
            --ngpu=1 \
            --phones-dict=dump/phone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use_gst=$use_gst \
            --use_vae=$use_vae || exit -1
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "copy files to model"
        cp dump/phone_id_map.txt exp/$model_name/phone_id_map.txt
        cp dump/speaker_id_map.txt exp/$model_name/speaker_id_map.txt
        cp dump/train/energy_stats.npy exp/$model_name/energy_stats.npy
        cp dump/train/pitch_stats.npy exp/$model_name/pitch_stats.npy
        cp dump/train/speech_stats.npy exp/$model_name/speech_stats.npy
    fi
fi


if [ ${fastspeech2} == False ] && [ ${multiple} == False ]; then
    echo "model: speedyspeech, single"

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        echo "duration"
        python tools/gen_duration_from_textgrid.py \
            --inputdir=data/TextGrid \
            --output=data/durations.txt \
            --config=train/conf/speedyspeech/default.yaml || exit -1
    fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "preprocess"
        python train/exps/speedyspeech/preprocess.py \
            --dataset=other \
            --rootdir=data/ \
            --dumpdir=dump \
            --dur-file=data/durations.txt \
            --config=train/conf/speedyspeech/default.yaml \
            --num-cpu=16 \
            --cut-sil=True \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "compute statistics"
        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="feats" \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo "normalize"
        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --dumpdir=dump/train/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1

        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/dev/raw/metadata.jsonl \
            --dumpdir=dump/dev/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1

        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/test/raw/metadata.jsonl \
            --dumpdir=dump/test/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] ; then
        echo "train"
        python train/exps/speedyspeech/train.py \
            --train-metadata=dump/train/norm/metadata.jsonl \
            --dev-metadata=dump/dev/norm/metadata.jsonl \
            --config=train/conf/speedyspeech/default.yaml \
            --output-dir=exp/$model_name \
            --ngpu=1 \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "copy files to model"
        cp dump/phone_id_map.txt exp/$model_name/phone_id_map.txt
        cp dump/speaker_id_map.txt exp/$model_name/speaker_id_map.txt
        cp dump/tone_id_map.txt exp/$model_name/tone_id_map.txt
        cp dump/train/feats_stats.npy exp/$model_name/feats_stats.npy
    fi
fi


if [ ${fastspeech2} == False ] && [ ${multiple} == True ]; then
    echo "model: speedyspeech, multiple"

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        echo "duration"
        python tools/gen_duration_from_textgrid.py \
            --inputdir=data/TextGrid \
            --output=data/durations.txt \
            --config=train/conf/speedyspeech/default_multi.yaml || exit -1
    fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "preprocess"
        python train/exps/speedyspeech/preprocess.py \
            --dataset=other \
            --rootdir=data/ \
            --dumpdir=dump \
            --dur-file=data/durations.txt \
            --config=train/conf/speedyspeech/default_multi.yaml \
            --num-cpu=16 \
            --cut-sil=True \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "compute statistics"
        python tools/compute_statistics.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --field-name="feats" \
            --use-relative-path=True|| exit -1
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo "normalize"
        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/train/raw/metadata.jsonl \
            --dumpdir=dump/train/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1

        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/dev/raw/metadata.jsonl \
            --dumpdir=dump/dev/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1

        python train/exps/speedyspeech/normalize.py \
            --metadata=dump/test/raw/metadata.jsonl \
            --dumpdir=dump/test/norm \
            --stats=dump/train/feats_stats.npy \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] ; then
        echo "train"
        python train/exps/speedyspeech/train.py \
            --train-metadata=dump/train/norm/metadata.jsonl \
            --dev-metadata=dump/dev/norm/metadata.jsonl \
            --config=train/conf/speedyspeech/default_multi.yaml \
            --output-dir=exp/$model_name \
            --ngpu=1 \
            --phones-dict=dump/phone_id_map.txt \
            --tones-dict=dump/tone_id_map.txt \
            --speaker-dict=dump/speaker_id_map.txt \
            --use-relative-path=True || exit -1
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        echo "copy files to model"
        cp dump/phone_id_map.txt exp/$model_name/phone_id_map.txt
        cp dump/speaker_id_map.txt exp/$model_name/speaker_id_map.txt
        cp dump/tone_id_map.txt exp/$model_name/tone_id_map.txt
        cp dump/train/feats_stats.npy exp/$model_name/feats_stats.npy
    fi
fi
