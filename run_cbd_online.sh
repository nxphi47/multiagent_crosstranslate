#!/usr/bin/env bash

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}
echo "num gpu: ${NUM_GPU}"


export seed="${seed:-12}"
export exp="${exp:-unsup_enfr}"
export data="${data:-en-fr}"
export src="${src:-en}"
export tgt="${tgt:-fr}"

export dump_dir=./dumped/${exp}-s${seed}
mkdir -p ${dump_dir}
export data_dir=./data/processed/${data}

# Params
export bsz="${bsz:-32}"
export max_epoch="${max_epoch:-30}"
#export max_epoch_pretrain="${max_epoch_pretrain:-$max_epoch}"
export max_epoch_pretrain="${max_epoch_pretrain:-1000}"
export emb_dim="${emb_dim:-1024}"
export n_layers="${n_layers:-6}"
export n_heads="${n_heads:-8}"
export dropout="${dropout:-0.1}"
export attention_dropout="${attention_dropout:-0.1}"
export gelu_activation="${gelu_activation:-true}"
# unmt params
export word_shuffle="${word_shuffle:-3}"
export word_dropout="${word_dropout:-0.1}"
export word_blank="${word_blank:-0.1}"
export word_mass="${word_mass:-0.5}"
export lambda_ae="${lambda_ae:-0:1,100000:0.1,300000:0}"
#export lambda_mass="${lambda_ae:-0:1,100000:0.1,300000:0}"

export tokens_per_batch="${tokens_per_batch:-2000}"
export tokens_per_batch_mlm="${tokens_per_batch_mlm:-2000}"
export upfreq="${upfreq:-1}"
export log_per_iter="${log_per_iter:-10}"
export eval_per_steps="${eval_per_steps:--1}"


export unmt_seed="${unmt_seed:-2}"

export nbest="${nbest:-3}"
export beam="${beam:-5}"
export lenpen="${lenpen:-1}"
export eval_only="${eval_only:-false}"
export fast_beam="${fast_beam:-true}"

export stop="${stop:-valid_en-fr_mt_bleu,10}"
export valmetrics="${valmetrics:-valid_en-fr_mt_bleu}"
#export stop="${stop:-valid_de-en_mt_bleu,10}"
#export valmetrics="${valmetrics:-valid_de-en_mt_bleu}"
export master="${master:-1234}"
export is_sentencepiece="${is_sentencepiece:-false}"

export split_data="${split_data:-false}"

export mbeam_ae="${mbeam_ae:-false}"
export mbeam_ae_epoch="${mbeam_ae_epoch:-5}"
export fast_beam_epoch="${fast_beam_epoch:-2}"
export beam_sample_temp="${beam_sample_temp:--1}"
export sampling_topp="${sampling_topp:--1}"
export sample_topn="${sample_topn:--1}"
export sample_temperature="${sample_temperature:-1}"
export diverse_beam_groups="${diverse_beam_groups:--1}"
export diverse_beam_strength="${diverse_beam_strength:--1}"
export initial_beam_epoch="${initial_beam_epoch:--1}"
export sample_topn_temp="${sample_topn_temp:--1}"
export sample_topn_replacement="${sample_topn_replacement:-false}"

export sec_bt_epoch="${sec_bt_epoch:--1}"
export mbeam_size="${mbeam_size:--1}"
export macd_version="${macd_version:-1}"
export arch="${arch:-mlm}"

export ae_steps_s="${ae_steps_s:---ae_steps ${src},${tgt}}"

export optimizer="${optimizer:-adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001}"

echo "================== Data Before Start =================="
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "data = ${data}"
echo "dump_dir = ${dump_dir}"
echo "data_dir = ${data_dir}"
echo "seed = ${seed}"
echo "exp = ${exp}"
echo "eval_only = ${eval_only}"
echo "length_penalty = ${lenpen}"
echo "fast_beam = ${fast_beam}"
echo "fast_beam_epoch = ${fast_beam_epoch}"
echo "beam_sample_temp = ${beam_sample_temp}"
echo "initial_beam_epoch = ${initial_beam_epoch}"
echo "======================================================="
# --------- PRETRAIN MLM --------------------

export pretrain_log=${dump_dir}/pretrain.log
export pretrain_name=pretrain-${exp}-s${seed}1
export pretrain_seed=${seed}1

# pretrain or not
export ispretrain="${ispretrain:-1}"
export PYTHONWARNINGS="ignore"
export dis=1

# --------- PRETRAIN MLM --------------------

export eval="${eval:-0}"

export train_log=${dump_dir}/train.log
export train_name=train
export train_seed=${seed}${unmt_seed}

export mass="${mass:-0}"

export python_command="train_online_cbd.py"

export pretrain_path="${pretrain_path:-}"
export reload_model_s="${reload_model_s:-}"
export xlm_path="${xlm_path:-}"
export mass_path="${mass_path:-}"
export label_smoothing="${label_smoothing:-0}"

export src_lang="${src_lang:-en}"
export tgt_lang="${tgt_lang:-fr}"

export filter_bleu="${filter_bleu:-0}"

#export path_s="--reload_model_xlm ${xlm_path},${xlm_path} --reload_model_mass ${mass_path},${mass_path} "
echo "Pretrained path = ${pretrain_path}"
echo "is_sentencepiece = ${is_sentencepiece}"

export NGPU=${NUM_GPU}

#--sample_temperature ${sample_temperature}   \
#export total_command="python3 ${python_command}  \
export total_command="python3 -u -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=${master} ${python_command}  \
--exp_name ${train_name} \
--exp_id s${train_seed} \
--seed ${train_seed} \
--dump_path ${dump_dir}  \
--data_path ${data_dir}  \
--is_sentencepiece ${is_sentencepiece}  \
--filter_bleu ${filter_bleu}  \
--lgs "${src}-${tgt}"   \
--src_lang ${src_lang} \
--tgt_lang ${tgt_lang} \
--bt_steps "${src}-${tgt}-${src},${tgt}-${src}-${tgt}" \
--reload_model_xlm ${xlm_path},${xlm_path} --reload_model_mass ${mass_path},${mass_path} ${reload_model_s} \
\
--beam_size ${beam}   \
--mbeam_size ${mbeam_size}   \
--nbest ${nbest}   \
--sample_topn ${sample_topn}   \
--fast_beam ${fast_beam}   \
--fast_beam_epoch ${fast_beam_epoch}   \
--label_smoothing ${label_smoothing}   \
--macd_version ${macd_version}   \
--arch ${arch}   \
\
--eval_only ${eval_only}   \
--length_penalty ${lenpen}   \
\
--word_shuffle ${word_shuffle} \
--word_dropout ${word_dropout} \
--word_blank ${word_blank} \
--word_mass ${word_mass} \
--lambda_ae ${lambda_ae} \
\
--split_data ${split_data} \
--encoder_only false  \
--emb_dim ${emb_dim}     \
--n_layers ${n_layers}    \
--n_heads ${n_heads}      \
--dropout ${dropout}     \
--attention_dropout ${attention_dropout} \
--gelu_activation ${gelu_activation}   \
--tokens_per_batch ${tokens_per_batch}   \
--batch_size ${bsz}   \
--bptt 256    \
--log_per_iter ${log_per_iter} \
--optimizer ${optimizer}  \
--epoch_size 200000 \
--eval_bleu true  \
--stopping_criterion ${stop}  \
--validation_metrics ${valmetrics} \
--max_epoch ${max_epoch} \
--fp16 1 --amp 1 --accumulate_gradients ${upfreq} \
| dd of=${train_log} "


eval ${total_command}


