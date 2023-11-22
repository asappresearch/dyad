import os
import pdb
import setuptools
import torch
import ipdb
import copy

from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig
from transformers.models.sparse_opt import SparseOPTForCausalLM
from transformers.models.sparse_opt.modeling_opt import SparseOPTConfig

from transformers import GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
from transformers import SparseGPTNeoXConfig
from transformers import SparseGPTNeoXForCausalLM

DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1

SPARSE_CONFIG_FILE = os.environ.get(
    'SPARSE_CONFIG_FILE',
    '/home/ubuntu/sparse_config'
)

def get_pythia_func(opt_model_size='160m'):
    model_name = f"EleutherAI/pythia-{opt_model_size}"
    config = GPTNeoXConfig.from_pretrained(model_name)
    model = GPTNeoXForCausalLM(config=config)
    return model

def get_sparse_pythia_func(opt_model_size='160m'):
    model_name = f"{SPARSE_CONFIG_FILE}/pythia-{opt_model_size}"
    config = SparseGPTNeoXConfig.from_pretrained(model_name)
    model = SparseGPTNeoXForCausalLM(config=config)
    return model

def get_opt_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    config = OPTConfig.from_pretrained(model_name)
    model = OPTForCausalLM(config=config)
    return model

def get_sparse_opt_func(opt_model_size='125m'):
    model_name = f"{SPARSE_CONFIG_FILE}/opt-{opt_model_size}"
    config = SparseOPTConfig.from_pretrained(model_name)
    model = SparseOPTForCausalLM(config=config)
    return model

def get_roberta_func(model_name="roberta-base", tokenizer=None):
    from transformers import RobertaConfig, RobertaForMaskedLM
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForMaskedLM(config)
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))
    return model
