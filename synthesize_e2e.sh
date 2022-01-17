set -e

# am_type=fastspeech2_aishell3
# am_model_name=fastspeech2_4people
# am_checkpoints=snapshot_iter_149990

# voc_type=hifigan_csmsc
# voc_model_name=hifigan_4people_finetuned
# voc_checkpoints=snapshot_iter_270000

am_type=speedyspeech_csmsc
am_model_name=speedyspeech_azi_nanami_1_9
am_checkpoints=snapshot_iter_63726

voc_type=hifigan_csmsc
voc_model_name=hifigan_azi_nanami_ft
voc_checkpoints=snapshot_iter_310000

fastspeech2=False
multiple=True
use_gst=False
use_vae=True
spk_id=175

if [ ${fastspeech2} == True ] && [ ${multiple} == True ]; then
    echo "model: fastspeech2, multiple"

    python train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/speech_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/finetune.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --speaker_dict=exp/${am_model_name}/speaker_id_map.txt \
        --ngpu=1 \
        --spk_id=${spk_id}
fi


if [ ${fastspeech2} == True ] && [ ${multiple} == False ]; then
    python train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/speech_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/finetune.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --ngpu=1

fi


if [ ${fastspeech2} == False ] && [ ${multiple} == True ]; then
    echo "model: speedyspeech, multiple"
    python3 train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/feats_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/finetune.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --tones_dict=exp/${am_model_name}/tone_id_map.txt \
        --speaker_dict=exp/${am_model_name}/speaker_id_map.txt \
        --ngpu=1 \
        --spk_id=${spk_id}
fi


if [ ${fastspeech2} == False ] && [ ${multiple} == False ]; then
    echo "model: speedyspeech, single"
    python3 train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/feats_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/finetune.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --tones_dict=exp/${am_model_name}/tone_id_map.txt \
        --ngpu=1
fi
