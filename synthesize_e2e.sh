set -e

am_type=fastspeech2_aishell3
am_model_name=fastspeech2_bili3_aishell3
am_checkpoints=snapshot_iter_179790

voc_type=hifigan_csmsc
voc_model_name=hifigan_azi_nanami
voc_checkpoints=snapshot_iter_115000

# am_type=speedyspeech_aishell3
# am_model_name=speedyspeech_azi_nanami_1_9
# am_checkpoints=snapshot_iter_63726

# voc_type=hifigan_csmsc
# voc_model_name=hifigan_azi_nanami_ft
# voc_checkpoints=snapshot_iter_310000

fastspeech2=True
multiple=True
use_style=True
use_gst=False
use_vae=False

spk_id=175
ngpu=0

str="ft"
if [[ $voc_model_name =~ $str ]]
then
    echo "voc_config is finetuned!"
    voc_config=finetune
else
    echo "voc_config is default!"    
    voc_config=default
fi

if [ ${fastspeech2} == True ] && [ ${multiple} == True ]; then
    echo "model: fastspeech2, multiple"

    python train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/speech_stats.npy \
        --pitch_stat=exp/${am_model_name}/pitch_stats.npy \
        --energy_stat=exp/${am_model_name}/energy_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/${voc_config}.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --speaker_dict=exp/${am_model_name}/speaker_id_map.txt \
        --ngpu=${ngpu} \
        --spk_id=${spk_id} \
        --use_gst=${use_gst} \
        --use_vae=${use_vae} \
        --use_style=${use_style} \
        --pitch_stat=exp/${am_model_name}/pitch_stats.npy \
        --energy_stat=exp/${am_model_name}/energy_stats.npy
fi


if [ ${fastspeech2} == True ] && [ ${multiple} == False ]; then
    python train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/speech_stats.npy \
        --pitch_stat=exp/${am_model_name}/pitch_stats.npy \
        --energy_stat=exp/${am_model_name}/energy_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/${voc_config}.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --ngpu=${ngpu} \
        --use_gst=${use_gst} \
        --use_vae=${use_vae} \
        --use_style=${use_style} \
        --pitch_stat=exp/${am_model_name}/pitch_stats.npy \
        --energy_stat=exp/${am_model_name}/energy_stats.npy

fi


if [ ${fastspeech2} == False ] && [ ${multiple} == True ]; then
    echo "model: speedyspeech, multiple"
    python3 train/exps/synthesize_e2e.py \
        --am=${am_type} \
        --am_config=exp/${am_model_name}/default_multi.yaml \
        --am_ckpt=exp/${am_model_name}/checkpoints/${am_checkpoints}.pdz \
        --am_stat=exp/${am_model_name}/feats_stats.npy \
        --voc=${voc_type} \
        --voc_config=pretrained_models/${voc_model_name}/${voc_config}.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --tones_dict=exp/${am_model_name}/tone_id_map.txt \
        --speaker_dict=exp/${am_model_name}/speaker_id_map.txt \
        --ngpu=${ngpu} \
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
        --voc_config=pretrained_models/${voc_model_name}/${voc_config}.yaml \
        --voc_ckpt=pretrained_models/${voc_model_name}/checkpoints/${voc_checkpoints}.pdz \
        --voc_stat=pretrained_models/${voc_model_name}/feats_stats.npy \
        --lang=zh \
        --text=sentences.txt \
        --output_dir=train/test_e2e \
        --inference_dir=train/inference \
        --phones_dict=exp/${am_model_name}/phone_id_map.txt \
        --tones_dict=exp/${am_model_name}/tone_id_map.txt \
        --ngpu=${ngpu}
fi
