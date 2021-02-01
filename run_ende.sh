#!/usr/bin/env bash

set -e


#---- version 3

# version 3 for de-en
export data=de-en
export src=de
export tgt=en
export src_lang=de
export tgt_lang=en
export split_data=true
export st=${src}-${tgt}
export stop=valid_${st}_mt_bleu,10
export valmetrics=valid_${st}_mt_bleu

export mlm_steps=" "
export log_per_iter=100
export max_epoch=50
export unmt_seed=2
export emb_dim=1024
export n_layers=6
export n_heads=8
export share_enc=-1
export share_dec=-1
export dropout=0.1
export attention_dropout=0.1
export gelu_activation=true
export bsz=32
export beam=5
export mbeam_size=-1
export seed=686
export fast_beam=true

export infer_single_model=false
export infer_single_mass=false
export xlmmass_reverse=false
#export xlm_path=dumped/pretrained_models/checkpoint.big.deen.pretxlm.s2546.pth
#export mass_path=dumped/pretrained_models/mass_ft_ende_1024.pth

export arch=mlm
#export pretrain_path=dumped/pretrained_models/${arch}_ende_1024.pth
export reload_model_s="--reload_model ${pretrain_path},${pretrain_path}"
#export reload_model_s="  "

export nbest=0
export seed=2347
export macd_version=3

export exp=${data}_unsup_cbd_nomb_onl_pret${arch}_ver${macd_version}_${src_lang}${tgt_lang}_${nbest}b${beam}

export label_smoothing=0
export max_epoch=60
export tokens_per_batch=2200
#export tokens_per_batch=2000
export upfreq=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export optimizer=adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001,warmup_updates=1000
export optimizer=adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001
#export CUDA_LAUNCH_BLOCKING=1 && bash train_online_macd.sh
unset CUDA_LAUNCH_BLOCKING
bash train_online_macd.sh


