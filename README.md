# Multi-Agent Cross-Translated Diversification
## ICML Submission


These guidelines demonstrate the steps to run CBD on the WMT En-De

#### 0. Installation

```bash
./install.sh
pip install fairseq==0.8.0 --progress-bar off
```


#### 1. Prepare data

Following instructions from [MASS-paper](https://github.com/microsoft/MASS) to create WMT En-De dataset.

#### 2. Prepare pretrained model

Download XLM finetuned model (theta_1): [here](), save it to bash variable `export xlm_path=...`

Download MASS finetuned model (theta_2): [here](https://modelrelease.blob.core.windows.net/mass/mass_ft_ende_1024.pth), save it to `export mass_path=....`

Download XLM pretrained model (theta): [here](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth), save it to `export pretrain_path...`


#### 3. Run CBD model
```bash

# you may change the inputs in the file according to your context
bash run_ende.sh

```

