#!/usr/bin/env bash

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}
echo "num gpu: ${NUM_GPU}"

#edmlm_noae_unsup_enfr
export seed="${seed:-12}"
export exp="${exp:-unsup_base_enfr}"
export data="${data:-en-fr}"
export src="${src:-en}"
export tgt="${tgt:-fr}"

export dump_dir=./dumped/${exp}-s${seed}
mkdir -p ${dump_dir}
export data_dir=./data/processed/${data}
export vecname="${vecname:-train.concat.vec}"
export data_vec=${data_dir}/${vecname}

# Params
export bsz="${bsz:-32}"
export max_epoch="${max_epoch:-30}"
export max_epoch_pretrain="${max_epoch_pretrain:-1000}"
export emb_dim="${emb_dim:-512}"
export n_layers="${n_layers:-6}"
export n_heads="${n_heads:-8}"
export dropout="${dropout:-0.1}"
export attention_dropout="${attention_dropout:-0}"
export gelu_activation="${gelu_activation:-true}"
# unmt params
export word_shuffle="${word_shuffle:-3}"
export word_dropout="${word_dropout:-0.1}"
export word_blank="${word_blank:-0.2}"
export word_pred="${word_pred:-0.2}"
export edmlm_full="${edmlm_full:-false}"

export share_enc="${share_enc:-5}"
export share_dec="${share_dec:-5}"

export lambda_ae="${lambda_ae:-0:1,100000:0.1,300000:0}"
export bt_sync="${bt_sync:-1000}"
export tokens_per_batch="${tokens_per_batch:-2000}"
export upfreq="${upfreq:-2}"


export PYTHONWARNINGS="ignore"

echo "================== Data Before Start =================="
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "data = ${data}"
echo "dump_dir = ${dump_dir}"
echo "data_dir = ${data_dir}"
echo "seed = ${seed}"
echo "exp = ${exp}"
echo "======================================================="

export pretrain_log=${dump_dir}/pretrain.log
export pretrain_name=pretrain-${exp}-s${seed}1
export pretrain_seed=${seed}1


#export pretrain_path=${dump_dir}/${pretrain_name}/s${pretrain_seed}/checkpoint.pth
export train_log=${dump_dir}/train.log
export train_name=train-${exp}-s${seed}2
export train_seed=${seed}2
export ae_steps="${ae_steps:-${src},${tgt}}"
#export mlm_steps="${mlm_steps:-${src},${tgt}}"
#export reload_model_train="${pretrain_path},${pretrain_path}"

export reload_emb="${reload_emb:-$data_vec}"
export reload_checkpoint="${reload_checkpoint:-}"

export optim="${optim:-adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001}"
export log_per_iter="${log_per_iter:-100}"
export master="${master:-1234}"

export stop="${stop:-valid_en-fr_mt_bleu,10}"
export valmetrics="${valmetrics:-valid_en-fr_mt_bleu}"


echo "reload_emb path = '${reload_emb}'"
echo "reload_checkpoint path = '${reload_checkpoint}'"
echo "ae_steps = '${ae_steps}'"
echo "mlm_steps = '${mlm_steps}'"
echo "edmlm_full = '${edmlm_full}'"
echo "word_pred = '${word_pred}'"
echo "optim = '${optim}'"
echo "bt_sync = '${bt_sync}'"
echo "stop = '${stop}'"
echo "valmetrics = '${valmetrics}'"
echo "master = '${master}'"
echo "----------------------------------------------------"


export NGPU=${NUM_GPU}; python -W ignore -u -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=${master}  train.py \
--exp_name ${train_name} \
--exp_id s${train_seed} \
--seed ${train_seed} \
--dump_path ${dump_dir}  \
--data_path ${data_dir}  \
--lgs "${src}-${tgt}"   \
\
--ae_steps "${ae_steps}"  \
--bt_steps "${src}-${tgt}-${src},${tgt}-${src}-${tgt}" \
--encoder_only false  \
\
--word_shuffle ${word_shuffle}   \
--word_dropout ${word_dropout}   \
--word_blank ${word_blank}    \
\
--word_pred ${word_pred} \
\
--reload_emb ${reload_emb} \
\
--share_enc ${share_enc} \
--share_dec ${share_dec} \
\
--lambda_ae ${lambda_ae} \
--bt_sync ${bt_sync} \
\
--emb_dim ${emb_dim}     \
--n_layers ${n_layers}    \
--n_heads ${n_heads}      \
--dropout ${dropout}     \
--attention_dropout ${attention_dropout} \
--gelu_activation ${gelu_activation}   \
--tokens_per_batch ${tokens_per_batch}   \
--batch_size ${bsz}   \
--bptt 256    \
--optimizer ${optim}  \
--epoch_size 200000 \
--eval_bleu true  \
--log_per_iter ${log_per_iter} \
--stopping_criterion "${stop}"  \
--validation_metrics "${valmetrics}" \
--max_epoch ${max_epoch} \
--fp16 1 --amp 1 \
--accumulate_gradients ${upfreq} | dd of=${train_log}






