# Multi-Agent Cross-Translated Diversification
## NeurIPS Submission

These guidelines demonstrate the steps to train multiple unsupervised MT agents and then generate synthetic parallel data. 
After that, we run a supervised MT as the final model. 

#### 0. Installation

```bash
./install.sh
pip install fairseq==0.8.0 --progress-bar off
```


#### 1. Prepare data
```bash

./get-data-nmt.sh --src en --tgt fr --reload_codes codes_enfr --reload_vocab vocab_enfr
./get-data-nmt-build-fasttext.sh --src en --tgt fr --data en-fr
 
```

#### 2. Run 2 UMT models
```bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# train 1st model
export seed=12
bash train_base_unmt.sh

# train 2nd model
export seed=13
bash train_base_unmt.sh

```

#### 3. Generate Synthetic data

```bash
export CUDA_VISIBLE_DEVICES=0

export seeds=31
export bt_steps=en-fr-en
export s1=12
export s2=13
export path=./dumped/unsup_base_enfr-s${s1}/train-unsup_base_enfr-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/unsup_base_enfr-s${s2}/train-unsup_base_enfr-s${s2}2/s${s2}2/checkpoint.pth
export exp=macd_gen_base_wmt_enfr_1213
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=8000
export infer_name=macd.${s1}-${s2}

bash prepare_synthetic_data.sh

export CUDA_VISIBLE_DEVICES=1

export seeds=31
export bt_steps=fr-en-en
export s1=12
export s2=13
export path=./dumped/unsup_base_enfr-s${s1}/train-unsup_base_enfr-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/unsup_base_enfr-s${s2}/train-unsup_base_enfr-s${s2}2/s${s2}2/checkpoint.pth
export exp=macd_gen_base_wmt_enfr_1213
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=8000
export infer_name=macd.${s1}-${s2}

bash prepare_synthetic_data.sh

export CUDA_VISIBLE_DEVICES=2

export seeds=31
export bt_steps=en-fr-en
export s1=13
export s2=12
export path=./dumped/unsup_base_enfr-s${s1}/train-unsup_base_enfr-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/unsup_base_enfr-s${s2}/train-unsup_base_enfr-s${s2}2/s${s2}2/checkpoint.pth
export exp=macd_gen_base_wmt_enfr_1213
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=8000
export infer_name=macd.${s1}-${s2}

bash prepare_synthetic_data.sh

export CUDA_VISIBLE_DEVICES=3

export seeds=31
export bt_steps=fr-en-en
export s1=13
export s2=12
export path=./dumped/unsup_base_enfr-s${s1}/train-unsup_base_enfr-s${s1}2/s${s1}2/checkpoint.pth
export parapath=./dumped/unsup_base_enfr-s${s2}/train-unsup_base_enfr-s${s2}2/s${s2}2/checkpoint.pth
export exp=macd_gen_base_wmt_enfr_1213
mkdir -p ./dumped/${exp}
export reload_para_model=${parapath},${parapath}
export reload_model=${path},${path}
export tokens_per_batch=8000
export infer_name=macd.${s1}-${s2}

bash prepare_synthetic_data.sh

```


#### 4. Combine the dataset
```bash

cd ./dumped/macd_gen_base_wmt_enfr_1213-s31/train-macd_gen_base_wmt_enfr_1213-s312/s312/hypotheses/

export infername=macd.12-13.rank0
cat ${infername}*.en-fr-en.train.b5.lp1.0.src.raw.en ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.en ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.en ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.en > ${infername}.cat.en
cat ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.fr ${infername}*.fr-en-fr.train.b5.lp1.0.src.raw.fr ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.fr ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.fr > ${infername}.cat.fr

export infername=macd.13-12.rank0
cat ${infername}*.en-fr-en.train.b5.lp1.0.src.raw.en ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.en ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.en ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.en > ${infername}.cat.en
cat ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.fr ${infername}*.fr-en-fr.train.b5.lp1.0.src.raw.fr ${infername}*.en-fr-en.train.b5.lp1.0.hyp.raw.fr ${infername}*.fr-en-fr.train.b5.lp1.0.hyp.raw.fr > ${infername}.cat.fr

cat macd.12-13.rank0.cat.en macd.13-12.rank0.cat.en > all1213.macd.ranks.infer0.cat.en
cat macd.12-13.rank0.cat.fr macd.13-12.rank0.cat.fr > all1213.macd.ranks.infer0.cat.fr

cd ../../../../../../


```

#### 5. Prepare Fairseq dataset
```bash

export trainpref=multiagent_crosstranslate/dumped/macd_gen_base_wmt_enfr_1213-s31/train-macd_gen_base_wmt_enfr_1213-s312/s312/hypotheses/all1213.macd.ranks.infer0.cat
export testpref=multiagent_crosstranslate/data/processed/en-fr/test
export validpref=multiagent_crosstranslate/data/processed/en-fr/valid
export destdir=translate_enfr_unmt_2stage_base_aug_wmt_bpe60k
fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref ${trainpref} \
  --testpref ${testpref} \
  --validpref ${validpref} \
  --destdir ${destdir} \
  --nwordssrc 0 --nwordstgt 0 \
  --joined-dictionary \
  --workers 48 

```


#### 6. Run supervised MT 

Run supervised MT following big Transformer in [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt)
