# Environment

Python 3.9, transformer package in huggingface, and datasets package in huggingface.

And also install: https://github.com/chengxuz/pt_framework

Install the current repo using `pip install .` or `pip install -e .`.

## Where to put data

First, define the environment variable `BABYLM_ROOT_DIR` to be where your models and data will live.
The downloaded data should be put at `${BABYLM_ROOT_DIR}/datasets/` so that this folder contains the following four subfolders: `babylm_100M`, `babylm_10M`, `babylm_dev`, and `babylm_test`. Note that the T5 training script expects .txt file inputs, so we create a single dev file by running this command in the `${BABYLM_ROOT_DIR}/datasets/babylm_dev/` folder: `cat *.dev > babylm_dev.txt`.
The trained models will be put at `${BABYLM_ROOT_DIR}/models/` and the records will be put at `${BABYLM_ROOT_DIR}/model_recs/`.

# Training Command

## OPT-125M
Run the following command under the `scripts` folder.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_strict.py:opt125m_s1"
```

This command will load a training setting specified by function `opt125m_s1` at `src/babylm_baseline_train/configs/BabyLM/exp_strict.py`.

## RoBERTa-Base
Run the following command under the `scripts` folder.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_strict_mask.py:roberta_s1"
```

## T5-Base
Run the following command under the `scripts` folder.
```
./train_t5_babylm.sh
```
Note that this training script uses a different backend than the OPT and RoBERTa models. This script is a slightly modified version of the `flax` T5 pre-training script from huggingface; the original [lives here](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling).

# Where important parameters are defined

Learning rate schedule is defined at function `get_learning_rate_params` in script `basic_param_setter.py` under `src/babylm_baseline_train` folder.

Optimizer is in the `scripts/general_train.py` script inside the `get_key_params` funciton.

# How to load the pretrained models

See the functions in `src/babylm_baseline_train/models/ckpt_loader.py`.

# Questions?

Feel free to open issues here. Or just contact us through Slack/emails.
