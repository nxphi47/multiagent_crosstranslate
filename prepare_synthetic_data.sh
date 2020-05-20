#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}
echo "num gpu: ${NUM_GPU}"

export seed="${seed:-12}"
export exp="${exp:-unsup_enfr}"
export data="${data:-en-fr}"
export src="${src:-en}"
export tgt="${tgt:-fr}"

echo "Generate data and combine: seeds: ${seed}"

export data_dir=./data/processed/${data}
export max_epoch="${max_epoch:-30}"
export emb_dim="${emb_dim:-512}"
export n_layers="${n_layers:-6}"
export n_heads="${n_heads:-8}"
export dropout="${dropout:-0.1}"
export attention_dropout="${attention_dropout:-0.1}"
export gelu_activation="${gelu_activation:-true}"
# unmt params
export word_shuffle="${word_shuffle:-3}"
export word_dropout="${word_dropout:-0.1}"
export word_blank="${word_blank:-0.1}"
export lambda_ae="${lambda_ae:-0:1,100000:0.1,300000:0}"
export tokens_per_batch="${tokens_per_batch:-2000}"
export bsz="${bsz:-32}"
export upfreq="${upfreq:-1}"
export beam="${beam:-5}"
export lenpen="${lenpen:-1}"

export share_enc="${share_enc:-5}"
export share_dec="${share_dec:-5}"

export mt_steps="${mt_steps:-$src-$tgt}"
export bt_steps="${bt_steps:-$src-$tgt-$src}"

export bt2infer="${bt2infer:-1}"
export infer_2stage_same_model="${infer_2stage_same_model:-false}"
export order_descending="${order_descending:-false}"
export split_data="${split_data:-false}"
export infer_name="${infer_name:-bt2inferout}"

export master="${master:-1234}"


export dump_dir=./dumped/${exp}-s${seed}
export train_name=train-${exp}-s${seed}2
export train_seed=${seed}2
export train_path=${dump_dir}/${train_name}/s${train_seed}/checkpoint.pth
export output_path=${dump_dir}/${train_name}/s${train_seed}/infer.train
export log_path=${dump_dir}/${train_name}/s${train_seed}/infer.tee.log

export reload_para_model="${reload_para_model:-notfound}"
export reload_model="${reload_model:-notfound}"

echo "================== Info Before Start =================="
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "data = ${data}"
echo "dump_dir = ${dump_dir}"
echo "data_dir = ${data_dir}"
echo "seed = ${seed}"
echo "exp = ${exp}"
echo "mt_steps = ${mt_steps}"
echo "bt_steps = ${bt_steps}"
echo "master = ${master}"
echo "train_path = ${train_path}"
echo "train_name = ${train_name}"
echo "train_seed = ${train_seed}"
echo "reload_para_model = ${reload_para_model}"
echo "reload_model = ${reload_model}"
echo "infer_2stage_same_model = ${infer_2stage_same_model}"
echo "======================================================="


export PYTHONWARNINGS="ignore"
export NGPU=${NUM_GPU}

export distributed="${distributed:-0}"


if [ ${distributed} -eq 1 ]; then
    export traincom="python3 -u -m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${master}  train_generate_macd.py "
else
    export traincom="python3 -u train_generate_macd.py "
fi

echo "---> generate 2 stage BT translation ---------------------------------"

export command="${traincom} \
--infer_train true \
--infer_2stage_train true \
--infer_2stage_same_model ${infer_2stage_same_model} \
--order_descending ${order_descending} \
--split_data ${split_data} \
\
--exp_name ${train_name} \
--exp_id s${train_seed} \
--seed ${train_seed} \
--dump_path ${dump_dir}  \
\
--data_path ${data_dir}  \
--infer_name ${infer_name}  \
\
--lgs "${src}-${tgt}"   \
--bt_steps ${bt_steps} \
--reload_para_model ${reload_para_model} \
--reload_model ${reload_model} \
\
--share_enc ${share_enc} \
--share_dec ${share_dec} \
\
--lambda_ae ${lambda_ae} \
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
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  \
--epoch_size 200000 \
--eval_bleu true  \
--stopping_criterion 'valid_en-fr_mt_bleu,10'  \
--validation_metrics 'valid_en-fr_mt_bleu' \
--max_epoch ${max_epoch} \
--fp16 1 --amp 1 \
--beam_size ${beam} \
--length_penalty ${lenpen} \
--accumulate_gradients ${upfreq} "

echo "--------------------"
echo $command
echo "--------------------"

eval ${command}


echo "---------->>>>>>>>>>>> FINISH Seed ${seed}"


export check=1

if [ $check -eq 0 ]; then

echo "nothing"



/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.en-fr.train.b1.lp1.en
/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.en-fr.train.b1.lp1.en
/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.en-fr.train.b1.lp1.en

export ori=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.en-fr.train.b1.lp1
export hypos=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.fr-en.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.en-fr.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.fr-en.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.en-fr.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.fr-en.train.b1.lp1
echo $hypos
echo $ori
python -u fi2_fairseq/combine_corpus.py --src en --tgt fr --ori $ori --hypos $hypos --dir raw_data/unmt_wmt14_aug_b1 --out train >> combine.hypo.unmt.b1.log 2>&1


python -u fi2_fairseq/combine_corpus.py --src en --tgt fr --ori $ori --hypos $hypos --dir raw_data/unmt_wmt14_aug_b1 --out train | tee combine.wmt14enfr.s2r1.log

fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref raw_data/unmt_wmt14_aug_b5lp1/train \
  --destdir data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_b5lp1 \
  --nwordssrc 0 --nwordstgt 0 --workers 16 \
  --srcdict data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_b5lp1/dict.en.txt \
  --tgtdict data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_b5lp1/dict.fr.txt >> preproc.enfr.aug.unmt.s3.log 2>&1



export ori=/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer3.en-fr.train.b5.lp1
export hypos=/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer3.fr-en.train.b5.lp1
python -u fi2_fairseq/combine_corpus.py --src en --tgt fr --ori $ori --hypos $hypos --dir raw_data/unmt_wmt14_aug_b5lp1 --out train



# no pretrained

#export ori=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.en-fr.train.b1.lp1
#export hypos=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.fr-en.train.b1.lp1
export ori=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.en-fr.train.b1.lp1
export hypos=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.fr-en.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.en-fr.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.fr-en.train.b1.lp1
echo $hypos
echo $ori
python -u fi2_fairseq/combine_corpus.py --src en --tgt fr --ori $ori --hypos $hypos --dir raw_data/unmt_wmt14_aug_b1_nopretrain_1213 --out train >> combine.hypo.unmt.b1.1213.log 2>&1



fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref raw_data/unmt_wmt14_aug_b1_nopretrain_1213/train \
  --destdir data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_s2_1213_b1lp1 \
  --nwordssrc 0 --nwordstgt 0 --workers 16 \
  --srcdict data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_s2_1213_b1lp1/dict.en.txt \
  --tgtdict data_fairseq_v2/translate_enfr_wmt14_bpe62k_from_unmt_s2_1213_b1lp1/dict.fr.txt >> preproc.enfr.aug.unmt.s2.1213.log 2>&1



# no pretrain

export CUDA_VISIBLE_DEVICES=6
export emb_dim=512
export n_layers=4
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0

export seeds=32
export exp=nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8
export data=en-fr
export src=en
export tgt=frd
export mt_steps=en-fr
export mt_steps=fr-en
export tokens_per_batch=4000
export bsz=32
export log_per_iter=100

bash prepare_merge_data.sh



# ---> change here as well -----------------
# TODO: generate 2-stage diversified data!
export CUDA_VISIBLE_DEVICES=3
export emb_dim=512
export n_layers=4
export n_layers=6
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0

export seeds=31
export exp=nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8
export exp=nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5
export data=en-fr
export src=en
export tgt=fr
export mt_steps=en-fr
export bt_steps=en-fr-en
#export mt_steps=fr-en
#export bt_steps=fr-en-fr

export log_per_iter=100
export bt2infer=1
export seeds=31
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s32/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s322/s322/checkpoint.pth

export seeds=32
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s31/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s312/s312/checkpoint.pth

export share_enc=5
export share_dec=5

export seeds=36
export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s35/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s352/s352/checkpoint.pth
export seeds=35
export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s36/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s362/s362/checkpoint.pth

export reload_para_model=${parapath},${parapath}
export tokens_per_batch=6000
export bsz=32

#change it to same model!
export infer_2stage_same_model=true
export infer_name=bt2same
bash prepare_merge_data.sh


# [en] -> fr1 => en2
# [fr] -> en1 => fr2

# [en] => fr1 -> en2
# [fr] => en1 -> fr2

# ------------------ todo: nopre_unsup_noedmlm_s1000

export CUDA_VISIBLE_DEVICES=1

export seeds=31
export exp=nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000
export data=en-fr
export src=en
export tgt=fr
export mt_steps=en-fr
export bt_steps=en-fr-en
export mt_steps=fr-en
export bt_steps=fr-en-fr

export log_per_iter=100
export bt2infer=1

export seeds=33
#export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s31/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s312/s312/checkpoint.pth
export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000-s33/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000-s332/s332/checkpoint.pth

export seeds=34
export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000-s34/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000-s342/s342/checkpoint.pth

export reload_para_model=${parapath},${parapath}
export tokens_per_batch=6000
export bsz=32
bash prepare_merge_data.sh





# combine
infer60.en-fr.train.b5.lp1.0.raw.en
infer60.en-fr.train.b5.lp1.0.raw.fr
infer60.fr-en.train.b5.lp1.0.raw.en

infer60.fr-en.train.b5.lp1.0.raw.fr

infer60.en-fr-en.train.b5.lp1.0.hyp.raw.en
infer60.en-fr-en.train.b5.lp1.0.hyp.raw.fr
infer60.en-fr-en.train.b5.lp1.0.src.raw.en

infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.en
infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.fr
infer60.fr-en-fr.train.b5.lp1.0.src.raw.fr


#en
cat infer60.en-fr.train.b5.lp1.0.raw.en infer60.fr-en.train.b5.lp1.0.raw.en infer60.en-fr-en.train.b5.lp1.0.hyp.raw.en infer60.en-fr-en.train.b5.lp1.0.src.raw.en infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.en infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.en >> infer60.all.cat.raw.en

cat infer60.en-fr.train.b5.lp1.0.raw.fr infer60.fr-en.train.b5.lp1.0.raw.fr infer60.en-fr-en.train.b5.lp1.0.hyp.raw.fr infer60.en-fr-en.train.b5.lp1.0.hyp.raw.fr infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.fr infer60.fr-en-fr.train.b5.lp1.0.src.raw.fr >> infer60.all.cat.raw.fr

#en right!
cat infer60.en-fr.train.b5.lp1.0.raw.en infer60.fr-en.train.b5.lp1.0.raw.en infer60.en-fr-en.train.b5.lp1.0.hyp.raw.en infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.en >> infer60.compact.cat.raw.en

cat infer60.en-fr.train.b5.lp1.0.raw.fr infer60.fr-en.train.b5.lp1.0.raw.fr infer60.en-fr-en.train.b5.lp1.0.hyp.raw.fr infer60.fr-en-fr.train.b5.lp1.0.hyp.raw.fr >> infer60.compact.cat.raw.fr


#export ori=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.en-fr.train.b1.lp1
#export hypos=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s11/train-unsup_enfr-s112/s112/hypotheses/infer10.fr-en.train.b1.lp1
export ori=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.en-fr.train.b1.lp1
export hypos=/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s12/train-unsup_enfr-s122/s122/hypotheses/infer19.fr-en.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.en-fr.train.b1.lp1
export hypos=${hypos}:/export/share/xnguyen/projects/nmt/fi_xlm/dumped/unsup_enfr-s13/train-unsup_enfr-s132/s132/hypotheses/infer20.fr-en.train.b1.lp1
echo $hypos
echo $ori
python -u fi2_fairseq/combine_corpus.py --src en --tgt fr --ori $ori --hypos $hypos --dir raw_data/unmt_wmt14_aug_b1_nopretrain_1213 --out train >> combine.hypo.unmt.b1.1213.log 2>&1


export d=fi_xlm/dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s31/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s312/s312/hypotheses
export trainpref=$d/infer60.all.cat.raw
export trainpref=$d/infer60.all.cat.raw.filtered
export testpref=fi_xlm/data/processed/en-fr/test.en-fr
export validpref=fi_xlm/data/processed/en-fr/valid.en-fr
export dict=${d}/vocab.en-fr
export out=data_fairseq_v2/translate_enfr_wmt14_bpe62k_unmt_2stage_s312_if60_fil

#python -u fi_xlm/filter_corpus.py --src en --tgt fr --ori ${trainpref}  --dir ${d} --out infer60.all.cat.raw.filtered
mkdir -p ${out}
cp -r $dict $out/dict.en.txt
cp -r $dict $out/dict.fr.txt
fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref ${trainpref} \
  --testpref ${testpref} \
  --validpref ${validpref} \
  --destdir ${out} \
  --nwordssrc 0 --nwordstgt 0 --workers 16 \
  --srcdict ${out}/dict.en.txt \
  --tgtdict ${out}/dict.fr.txt



import numpy as np
data  = "infer.train1.probs"
data  = "infer.train4.probs"
data  = "translate_ende_iwslt14_bpe32k/baseline_tfm_normal_share_mul3/model_1/infer_train_b1_lp1.probs"
data  = "translate_ende_iwslt14_bpe32k/baseline_tfm_normal_share_mul3/model_2/infer_train_b1_lp1.probs"
data  = "translate_ende_iwslt14_bpe32k/baseline_tfm_normal_share_mul3/model_3/infer_train_b1_lp1.probs"
data  = "translate_ende_aug_deenwende_iwslt14_bpe32k/baseline_tfm_normal_dr0_s4598_mul3/model_1/infer.train4.probs"
data  = "translate_deen_aug_endewdeen_iwslt14_bpe32k/baseline_tfm_normal_dr0_s3011_mul3/model_1/infer.train1.probs"
data  = "translate_deen_aug_endewdeen_iwslt14_bpe32k/baseline_tfm_normal_share_s301_mul3/model_1/infer.train1.probs"
data  = "translate_deen_aug_endewdeen_iwslt14_bpe32k/baseline_tfm_normal_share_s301_mul3/model_1/infer.train2.probs"

import numpy as np
data  = "translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_1/infer_train_b1_lp1.probs"
data  = "translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_2/infer_train_b1_lp1.probs"
with open(data) as f:
    lines = f.read().strip().split('\n')
    arr = []
    for l in lines:
        arr.extend([np.exp(float(x)) for x in l.split()])

print(sum(arr)/len(arr))

#aug ende:
#0.8390177737615656
#0.8561279231379255 (dr0)


#aug deen:
#0.85 (train1)
#0.88 (dr0) (train1)

#ende
#0.7520265568640321

#deen
#0.7670396010370656


# todo: redo-
import numpy as np
datas  = ["translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_1/infer_train_b1_lp1.probs"]
datas.append("translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_2/infer_train_b1_lp1.probs")
datas.append("translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_3/infer_train_b1_lp1.probs")
arr = []
for data in datas:
    with open(data) as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            arr.extend([np.exp(float(x)) for x in l.split()])


print(sum(arr)/len(arr))
# original deen: 0.7699932279232793


# TODO: check probs
export index=1
export beam=1
export lenpen=1
export problem=translate_deen_aug_endewdeen_iwslt14_bpe32k
export model_dir=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_share_s301_mul3/model_${index}
export model_dir=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_dr0_s3011_mul3/model_${index}
export best_file=$model_dir/checkpoint_best.pt
export data_dir=`pwd`/data_fairseq_v2/$problem
export gen_out=$model_dir/infer_train_testing
python fi2_fairseq/check_probs.py ${data_dir} --path ${best_file} --user-dir fi2_fairseq --gen-subset train4 --max-tokens 8192 --beam ${beam}  --lenpen ${lenpen}

# TODO: check probs
export index=1
export beam=1
export lenpen=1
export problem=translate_deen_aug_endewdeen_iwslt14_bpe32k
export model_dir=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_share_s301_mul3/model_${index}
export model_dir=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_dr0_s3011_mul3/model_${index}
export model_dir=`pwd`/train_fi_fairseq/$problem/big_tfm_baseline_dr0_s3011_mul3/model_${index}
export best_file=$model_dir/checkpoint_best.pt
export best_file=$model_dir/checkpoint_last.pt
export data_dir=`pwd`/data_fairseq_v2/$problem
export gen_out=$model_dir/infer_train_testing
python fi2_fairseq/check_probs.py ${data_dir} --path ${best_file} --user-dir fi2_fairseq --gen-subset train4 --max-tokens 4096 --beam ${beam}  --lenpen ${lenpen}


# todo: check prob for En-De
export index=1
export beam=1
export lenpen=1
export problem=translate_ende_aug_deenwende_iwslt14_bpe32k
#export best_file=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_share_s100_mul3/model_${index}/checkpoint_best.pt
export best_file=`pwd`/train_fi_fairseq/$problem/baseline_tfm_normal_dr0_s4598_mul3/model_${index}/checkpoint_last.pt
#export model_dir=`pwd`/train_fi_fairseq/$problem/big_tfm_baseline_dr0_s3011_mul3/model_${index}
#export best_file=$model_dir/checkpoint_best.pt
#export best_file=$model_dir/checkpoint_last.pt
export data_dir=`pwd`/data_fairseq_v2/$problem
export gen_out=$model_dir/infer_train_testing
python fi2_fairseq/check_probs.py ${data_dir} --path ${best_file} --user-dir fi2_fairseq --gen-subset train4 --max-tokens 4096 --beam ${beam}  --lenpen ${lenpen}


# check peer model on its own predictions De- En
export index=1
export train_id=4
export beam=1
export lenpen=1
export best_file=`pwd`/train_fi_fairseq/translate_deen_iwslt14_bpe32k_v3/baseline_tfm_normal_share_mul3/model_${index}/checkpoint_best.pt
export best_file=`pwd`/train_fi_fairseq/translate_deen_aug_endewdeen_iwslt14_bpe32k_conf/baseline_tfm_normal_deen_share_conf_s23098_mul1/model_1/checkpoint_best.pt
export data_dir=`pwd`/data_fairseq_v2/translate_deen_aug_endewdeen_iwslt14_bpe32k
export gen=${best_file}.train${train_id}.probs
python fi2_fairseq/check_probs.py ${data_dir} --path ${best_file} --user-dir fi2_fairseq --gen-subset train${train_id} --max-tokens 1024 --beam ${beam}  --lenpen ${lenpen} --report_out | dd of=${gen}



# check peer model on its own predictions for En-De
export index=2
export train_id=5
export beam=1
export lenpen=1
export best_file=`pwd`/train_fi_fairseq/translate_ende_iwslt14_bpe32k/baseline_tfm_normal_share_mul3/model_${index}/checkpoint_best.pt
export data_dir=`pwd`/data_fairseq_v2/translate_ende_aug_deenwende_iwslt14_bpe32k
python fi2_fairseq/check_probs.py ${data_dir} --path ${best_file} --user-dir fi2_fairseq --gen-subset train${train_id} --max-tokens 8192 --beam ${beam}  --lenpen ${lenpen}




# back-translation with IWSLT De-En
export problem=translate_enfr_wmt14_bpe62k_from_unmt_test
export problem=translate_enfr_wmt14_bpe62k_unmt_2stage_s312_if60_fil
#export problem=translate_deen_bt_aug_s3_r0_iwslt14_bpe32k
export CUDA_VISIBLE_DEVICES=6
export src_lang=en
export tgt_lang=fr
export updates=500000
export upfreq=8
export multiple=1
export seed_prefix=21
bash fi2_fairseq/run_multiple_baseline_base.sh





# baseline multiple
export seed_prefix=5467
export problem=translate_enfr_wmt14_bpe62k_unmt_2stage_s312_if60_fil
export arch=transformer_vaswani_wmt_en_fr_big
export dropout=0.1
export att_dropout=0
export tgt_lang=fr
export name=dfnew_b3584_lr5x
export updates=90000
export avg_num=5
export lr=0.001
export CUDA_VISIBLE_DEVICES=6,7 && export upfreq=32 && export maxtokens=7100 && bash fi2_fairseq/run_multiple_baseline_big.sh


#{'dump_path': './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212',
# 'exp_name': 'train-unsup_enfr-s212',
# 'save_periodic': 0,
# 'exp_id': 's212',
# 'fp16': True, 'amp': 1,
# 'encoder_only': False,
# 'emb_dim': 1024, 'n_layers': 6, 'n_heads': 8, 'dropout': 0.1, 'attention_dropout': 0.1, 'gelu_activation': True, 'share_inout_emb': True, 'sinusoidal_embeddings': False, 'use_lang_emb': True, 'use_memory': False, 'asm': False, 'context_size': 0, 'word_pred': 0.15, 'sample_alpha': 0, 'word_mask_keep_rand': '0.8,0.1,0.1', 'word_shuffle': 3.0, 'word_dropout': 0.1, 'word_blank': 0.1, 'data_path': './data/processed/en-fr', 'lgs': 'en-fr', 'max_vocab': -1, 'min_count': 0, 'lg_sampling_factor': -1, 'bptt': 256, 'max_len': 100, 'group_by_size': True, 'batch_size': 32, 'max_batch_size': 0, 'tokens_per_batch': 2000, 'split_data': False, 'optimizer': 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001', 'clip_grad_norm': 5, 'epoch_size': 200000, 'max_epoch': 30, 'stopping_criterion': 'valid_en-fr_mt_bleu,10', 'validation_metrics': 'valid_en-fr_mt_bleu', 'accumulate_gradients': 1, 'lambda_mlm': 1.0, 'lambda_clm': 1.0, 'lambda_pc': 1.0, 'lambda_ae': 0.57799, 'lambda_mt': 1.0, 'lambda_bt': 1.0, 'clm_steps': [], 'mlm_steps': [], 'mt_steps': [], 'ae_steps': ['en', 'fr'], 'bt_steps': [('en', 'fr', 'en'), ('fr', 'en', 'fr')], 'pc_steps': [], 'reload_emb': '', 'reload_model': './dumped/unsup_enfr-s21/pretrain-unsup_enfr-s211/s211/checkpoint.pth,./dumped/unsup_enfr-s21/pretrain-unsup_enfr-s211/s211/checkpoint.pth', 'reload_checkpoint': '', 'beam_size': 1, 'length_penalty': 1, 'early_stopping': False, 'eval_bleu': True, 'eval_only': False, 'infer_train': False, 'debug_train': False, 'debug_slurm': False, 'debug': False, 'local_rank': 0, 'master_port': -1, 'seed': 212, 'langs': ['en', 'fr'], 'id2lang': {0: 'en', 1: 'fr'}, 'lang2id': {'en': 0, 'fr': 1}, 'n_langs': 2, 'bt_src_langs': ['en', 'fr'], 'mono_dataset': {'en': {'train': './data/processed/en-fr/train.en.pth', 'valid': './data/processed/en-fr/valid.en.pth', 'test': './data/processed/en-fr/test.en.pth'}, 'fr': {'train': './data/processed/en-fr/train.fr.pth', 'valid': './data/processed/en-fr/valid.fr.pth', 'test': './data/processed/en-fr/test.fr.pth'}}, 'para_dataset': {('en', 'fr'): {'valid': ('./data/processed/en-fr/valid.en-fr.en.pth', './data/processed/en-fr/valid.en-fr.fr.pth'), 'test': ('./data/processed/en-fr/test.en-fr.en.pth', './data/processed/en-fr/test.en-fr.fr.pth')}}, 'word_mask': 0.8, 'word_keep': 0.1, 'word_rand': 0.1, 'is_slurm_job': False, 'global_rank': 0, 'world_size': 8, 'n_gpu_per_node': 8, 'n_nodes': 1, 'node_id': 0, 'is_master': True, 'multi_node': False, 'multi_gpu': True, 'command': 'python train.py --local_rank=0 --exp_name \'train-unsup_enfr-s212\' --exp_id s212 --seed 212 --dump_path \'./dumped/unsup_enfr-s21\' --reload_model \'./dumped/unsup_enfr-s21/pretrain-unsup_enfr-s211/s211/checkpoint.pth,./dumped/unsup_enfr-s21/pretrain-unsup_enfr-s211/s211/checkpoint.pth\' --data_path \'./data/processed/en-fr\' --lgs \'en-fr\' --ae_steps \'en,fr\' --bt_steps \'en-fr-en,fr-en-fr\' --word_shuffle 3 --word_dropout \'0.1\' --word_blank \'0.1\' --lambda_ae \'0:1,100000:0.1,300000:0\' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout \'0.1\' --attention_dropout \'0.1\' --gelu_activation true --tokens_per_batch 2000 --batch_size 32 --bptt 256 --optimizer \'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001\' --epoch_size 200000 --eval_bleu true --stopping_criterion \'valid_en-fr_mt_bleu,10\' --validation_metrics \'valid_en-fr_mt_bleu\' --max_epoch 30 --fp16 1 --amp 1 --accumulate_gradients 1 --exp_id "s212"', 'n_words': 64139, 'bos_index': 0, 'eos_index': 1, 'pad_index': 2, 'unk_index': 3, 'mask_index': 5, 'pred_probs': tensor([0.8000, 0.1000, 0.1000]), 'mask_scores': array([0, 0, 0, ..., 1, 1, 1]), 'lambda_clm_config': None, 'lambda_mlm_config': None, 'lambda_pc_config': None, 'lambda_ae_config': [(0, 1.0), (100000, 0.1), (300000, 0.0)], 'lambda_mt_config': None, 'lambda_bt_config': None, 'hyp_path': './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212/hypotheses', 'ref_paths': {('fr', 'en', 'valid'): './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212/hypotheses/ref.fr-en.valid.txt', ('en', 'fr', 'valid'): './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212/hypotheses/ref.en-fr.valid.txt', ('fr', 'en', 'test'): './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212/hypotheses/ref.fr-en.test.txt', ('en', 'fr', 'test'): './dumped/unsup_enfr-s21/train-unsup_enfr-s212/s212/hypotheses/ref.en-fr.test.txt'}}


# TODO: Prepare merge data very big!
# ---> change here as well -----------------
export CUDA_VISIBLE_DEVICES=1
export emb_dim=1024
export n_layers=6
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0.1

export seeds=31
export exp=nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8
export exp=nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5
export data=en-fr
export src=en
export tgt=fr
export mt_steps=en-fr
export bt_steps=en-fr-en
export mt_steps=fr-en
export bt_steps=fr-en-fr

export log_per_iter=100
export bt2infer=1
export seeds=31
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s32/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s322/s322/checkpoint.pth

export seeds=32
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s31/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s312/s312/checkpoint.pth

export share_enc=5
export share_dec=5
export seeds=36
export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s35/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s352/s352/checkpoint.pth
#export seeds=35
#export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s36/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s362/s362/checkpoint.pth


export reload_para_model=${parapath},${parapath}
export tokens_per_batch=6000
export bsz=32


export infer_name=testinginfer
export order_descending=true
export split_data=true
bash prepare_merge_data.sh



# TODO: Factor=3 base enfr 35-36-37


# ---> change here as well -----------------
# TODO: generate 2-stage diversified data!
export CUDA_VISIBLE_DEVICES=3
export emb_dim=512
export n_layers=4
export n_layers=6
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0

export seeds=31
export exp=nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8
export exp=nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5
export data=en-fr
export src=en
export tgt=fr
export mt_steps=en-fr
export bt_steps=en-fr-en
#export mt_steps=fr-en
#export bt_steps=fr-en-fr

export log_per_iter=100
export bt2infer=1
export seeds=31
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s32/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s322/s322/checkpoint.pth

export seeds=32
export parapath=./dumped/nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s31/train-nopretrain_unsup_l4_enfr_1gpu_tpb1k_x8-s312/s312/checkpoint.pth

export share_enc=5
export share_dec=5

export seeds=36
#export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s35/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s352/s352/checkpoint.pth
#export seeds=35
#export parapath=./dumped/nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s36/train-nopretrain_unsup_enfr_1gpu_tpb1k_x8_bts1000_sh5-s362/s362/checkpoint.pth

export reload_para_model=${parapath},${parapath}
export tokens_per_batch=6000
export bsz=32

export parapath=./dumped/bt2stage_deen/s312.checkpoint.pth
export path=./dumped/bt2stage_deen/s322.checkpoint.pth
#export exp=bt2stage_deen_3231_out
export exp=bt2stage_base_enfr_353637

mkdir -p ./dumped/${exp}

export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=4000
export bsz=64

export tokens_per_batch=8000
export bsz=64

export tokens_per_batch=16000
export bsz=128

export order_descending=true

export split_data=false

# distributed
export CUDA_VISIBLE_DEVICES=2,3
export distributed=1
export split_data=true
export infer_name=bt2stagedis

#change it to same model!
#export infer_2stage_same_model=true
#export infer_name=bt2same
bash prepare_merge_data.sh



# generate
export CUDA_VISIBLE_DEVICES=0
export s1=37
export s2=35
export mt=en-fr
export bt=en-fr-en

export CUDA_VISIBLE_DEVICES=1
export s1=37
export s2=35
export mt=fr-en
export bt=fr-en-fr

export CUDA_VISIBLE_DEVICES=2
export s1=35
export s2=37
export mt=en-fr
export bt=en-fr-en

export CUDA_VISIBLE_DEVICES=3
export s1=35
export s2=37
export mt=fr-en
export bt=fr-en-fr


# generate
export CUDA_VISIBLE_DEVICES=4
export s1=37
export s2=36
export mt=en-fr
export bt=en-fr-en

export CUDA_VISIBLE_DEVICES=5
export s1=37
export s2=36
export mt=fr-en
export bt=fr-en-fr

export CUDA_VISIBLE_DEVICES=6
export s1=36
export s2=37
export mt=en-fr
export bt=en-fr-en

export CUDA_VISIBLE_DEVICES=7
export s1=36
export s2=37
export mt=fr-en
export bt=fr-en-fr




# --- TODO: IWSLT generate for de-en
export CUDA_VISIBLE_DEVICES=3
export emb_dim=512
export n_layers=5
export share_enc=4
export share_dec=4
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0

export seeds=31

export log_per_iter=100
export bt2infer=1

export data=iwslt-de-en
export src=de
export tgt=en
export mt_steps=de-en
export bt_steps=de-en-de
export mt_steps=en-de
export bt_steps=en-de-en

#export mt_steps=fr-en
#export bt_steps=fr-en-fr
export s1=51
export s2=53
#export s1=53
#export s2=51


export path=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
#export exp=bt2stage_deen_3231_out
echo $path
echo $parapath
export exp=bt2stage_iwslt_deen_51525354

mkdir -p ./dumped/${exp}

export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}

export tokens_per_batch=4000
export bsz=64
export tokens_per_batch=8000
export bsz=64
export tokens_per_batch=16000
export bsz=128

export order_descending=false
export split_data=false
export infer_name=bt2stage.${s1}-${s2}
bash prepare_merge_data.sh


# fixme: do with for loop here:
export CUDA_VISIBLE_DEVICES=2
export emb_dim=512
export n_layers=5
export share_enc=4
export share_dec=4
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0
export seeds=31
export log_per_iter=100
export bt2infer=1

export data=iwslt-de-en
export src=de
export tgt=en
#export s1=53
#export s2=52
export p1=54
export p2=53
# loop for n=4: 54-51, 54-52, 54-53
for bt in de-en-de en-de-en; do
    export bt_steps=$bt
    for s1 in $p1 $p2; do
        for s2 in $p2 $p1; do
            if [ $s1 == $s2 ]; then
                continue
            fi
            echo "Bt steps :$bt_steps"
            echo "s1: $s1"
            echo "s2: $s2"
            export path=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
            export parapath=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
            echo $path
            echo $parapath
            export exp=bt2stage_iwslt_deen_51525354
            mkdir -p ./dumped/${exp}
            export reload_para_model=${parapath},${parapath}
            export reload_model=${path},${path}
            export tokens_per_batch=16000
            export bsz=128
            export order_descending=false
            export split_data=false
            export infer_name=bt2stage.${s1}-${s2}
            bash prepare_merge_data.sh
        done
    done
done


#export bt_steps=de-en-de

#export mt_steps=de-en
#for bt in de-en-de en-de-en; do
#export bt_steps=$bt
export bt_steps=de-en-de
export bt_steps=en-de-en
echo "Bt steps :$bt_steps"
echo "s1: $s1"
echo "s2: $s2"
export path=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-de-en-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
echo $path
echo $parapath
export exp=bt2stage_iwslt_deen_51525354
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=16000
export bsz=128
export order_descending=false
export split_data=false
export infer_name=bt2stage.${s1}-${s2}
bash prepare_merge_data.sh
#done


#export mt_steps=en-de
#export bt_steps=en-de-en
for bt in de-en-de en-de-en; do
export bt_steps=$bt
for s1 in 53 52; do
    for s2 in 53 52; do
        if [ $s1 == $s2 ]; then
            continue
        fi
        echo "Bt steps :$bt_steps"
        echo "s1: $s1"
        echo "s2: $s2"
    done
done
done



# todo: Generate synthetic data for IWSLT en-fr
export CUDA_VISIBLE_DEVICES=2
export emb_dim=512
export n_layers=5
export share_enc=4
export share_dec=4
export n_heads=8
export upfreq=8
export dropout=0.1
export attention_dropout=0

export seeds=31
export log_per_iter=100
export bt2infer=1

export data=iwslt-en-fr
export src=en
export tgt=fr
#export s1=62
#export s2=63
export s1=63
export s2=62
export bt_steps=en-fr-en
#export s1=$ss1
#export s2=$ss2
#echo "Bt steps :$bt_steps"
#echo "s1: $s1"
#echo "s2: $s2"
#export path=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
#export parapath=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
#echo "path: $path"
#echo "parapath: $parapath"
#export exp=bt2stage_iwslt_enfr_61626364
#mkdir -p ./dumped/${exp}
#export reload_para_model=${parapath},${parapath}
#export reload_model=${path},${path}
#export tokens_per_batch=8000
#export bsz=64
#export order_descending=false
#export split_data=false
#export infer_name=bt2stage.${s1}-${s2}
#bash prepare_synthetic_data.sh

export p1=66
export p2=63
# loop for n=4: 54-51, 54-52, 54-53
for bt in en-fr-en fr-en-fr; do
    export bt_steps=$bt
    for s1 in $p1 $p2; do
        for s2 in $p2 $p1; do
            if [ $s1 == $s2 ]; then
                continue
            fi
            echo "Bt steps :$bt_steps"
            echo "s1: $s1"
            echo "s2: $s2"
            export path=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
            export parapath=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
            echo "path: $path"
            echo "parapath: $parapath"
            export exp=bt2stage_iwslt_enfr_61626364
            mkdir -p ./dumped/${exp}
            export reload_para_model=${parapath},${parapath}
            export reload_model=${path},${path}
            export tokens_per_batch=8000
            export bsz=64
            export order_descending=false
            export split_data=false
            export infer_name=bt2stage.${s1}-${s2}
            bash prepare_merge_data.sh
        done
    done
done

#export mt_steps=de-en
for bt in en-fr-en fr-en-fr; do
export bt_steps=$bt
#for ss1 in 61 63; do
#    for ss2 in 61 63; do
#        if [ $ss1 == $ss2 ]; then
#            continue
#        fi
export s1=$ss1
export s2=$ss2
echo "Bt steps :$bt_steps"
echo "s1: $s1"
echo "s2: $s2"
export path=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}/train-iwslt-en-fr-nopretrain_unsup_1gpu_32tpb2k_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
echo "path: $path"
echo "parapath: $parapath"
export exp=bt2stage_iwslt_enfr_61626364
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=8000
export bsz=64
export order_descending=false
export split_data=false
export infer_name=bt2stage.${s1}-${s2}
bash prepare_merge_data.sh
#    done
#done
done


#export mt_steps=en-de
#export bt_steps=en-de-en
for bt in de-en-de en-de-en; do
export bt_steps=$bt
for s1 in 53 52; do
    for s2 in 53 52; do
        if [ $s1 == $s2 ]; then
            continue
        fi
        echo "Bt steps :$bt_steps"
        echo "s1: $s1"
        echo "s2: $s2"
    done
done
done






# todo: Generate data for base model on 32Gb De-En
# 46-47 de-en-de : dgx 0,1 done
# 46-47 en-de-en : dgx 4,5 todo doing, but will fail
# 47-46 de-en-de : dgx 6,7 done
# 47-46 en-de-en : dgx 0,1 on going todo (300, 25000)
# 47-46 en-de-en ascending: dgx 6,7 on going todo (300, 25000)
export seeds=31
export data=de-en
export src=de
export tgt=en
export stop=valid_de-en_mt_bleu,10
export valmetrics=valid_de-en_mt_bleu
export vecname=train.concat.vec
export reload_emb=data/processed/de-en/train.concat.vec

export max_epoch=50
export emb_dim=512
export n_layers=6
export share_enc=5
export share_dec=5
export n_heads=8
export upfreq=8
export mlm_steps=" "
export lambda_ae=0:1,100000:0.1,400000:0
export dropout=0.1
export attention_dropout=0
export bt_sync=1000
export log_per_iter=100
export exp=nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5
export upfreq=1

export bt_steps=de-en-de
export CUDA_VISIBLE_DEVICES=4,5
export bt_steps=en-de-en
export s1=46
export s2=47

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=6,7
export bt_steps=de-en-de
export bt_steps=en-de-en
export s1=47
export s2=46

export path=./dumped/nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s1}/train-nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s2}/train-nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
echo "path: $path"
echo "parapath: $parapath"
export exp=bt2stage_base_wmt_deen_4647
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}

export tokens_per_batch=20000
export bsz=200
export tokens_per_batch=25000
export bsz=300

export order_descending=true
export split_data=false
export infer_name=bt2stage.${s1}-${s2}

export distributed=1
export split_data=true
export infer_name=bt2stagedis.${s1}-${s2}

bash prepare_merge_data.sh




# todo: Generate data for base model on 32Gb En-Ro
# 76-77 ro-en-ro : dgx 3
# 77-76 ro-en-ro : dgx 7
export seeds=31
#export seeds=99
export data=en-ro
export src=en
export tgt=ro
export stop=valid_en-ro_mt_bleu,10
export valmetrics=valid_en-ro_mt_bleu
export vecname=train.concat.vec
export reload_emb=data/processed/en-ro/train.concat.vec
export bt2infer=1

export max_epoch=50
export emb_dim=512
export n_layers=6
export share_enc=5
export share_dec=5
export n_heads=8
export upfreq=8
export mlm_steps=" "
export lambda_ae=0:1,100000:0.1,400000:0
export dropout=0.1
export attention_dropout=0
export bt_sync=1000
export log_per_iter=100
export exp=nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5
export upfreq=1

export bt_steps=ro-en-ro
export s1=76
export s2=77
#export s1=77
#export s2=76

export path=./dumped/nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s1}/train-nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s2}/train-nopretrain_unsup_deen_1gpu_tpb8k128b_x1_bts1000_sh5-s${s2}2/s${s2}2/checkpoint.pth
echo "path: $path"
echo "parapath: $parapath"
export exp=bt2stage_base_wmt_enro_7677-s31
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}

export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export tokens_per_batch=20000
export bsz=200
export tokens_per_batch=20000
export tokens_per_batch=10000
export bsz=150

export order_descending=false
#export order_descending=true
export split_data=false
export infer_name=bt2stage.${s1}-${s2}
export infer_name=placeholderbt2stage.${s1}-${s2}
export distributed=1
export split_data=true
export infer_name=bt2stagedis.${s1}-${s2}

bash prepare_merge_data.sh





fi











