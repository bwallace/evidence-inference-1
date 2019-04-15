import argparse
import copy
import os
import random
import sys
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import dill  

from os.path import abspath, dirname, join
# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import torch
import torch.nn as nn
from torch.nn import functional as F
from evidence_inference.models.utils import PaddedSequence
from evidence_inference.models.model_0 import *
from evidence_inference.experiments.model_0_paper_experiment import get_data
from evidence_inference.models.train_section_attn import train_section_attn

USE_CUDA = True

def main():
    """
    Load in data, and run the model.
    """
    train_Xy, val_Xy, test_Xy, inference_vectorizer = get_data(mode = 'minimal')
    
    
    nn_sec_attn = EvidenceInferenceSections(inference_vectorizer, 
                                              h_size=32,
                                              init_embeddings=None,
                                              init_wvs_path="embeddings/PubMed-w2v.bin",
                                              weight_tying=False,
                                              ICO_encoder="CBoW",
                                              article_encoder="GRU",
                                              condition_attention=True,
                                              tokenwise_attention=False,
                                              tune_embeddings=False,
                                              section_attn_embedding=32,
                                              use_attention_over_article_tokens=True, 
                                              recursive_encoding = False)
    
    results = train_section_attn(nn_sec_attn, 
                                 train_Xy, 
                                 val_Xy, 
                                 test_Xy, 
                                 inference_vectorizer, 
                                 epochs=25, 
                                 batch_size=1, 
                                 shuffle=True, 
                                 gen_big_sections = True,
                                 recursive_encoding = False,
                                 pretrain_tokens = False, 
                                 pretrain_sections = False)

    print(results[4]) 


if __name__ == '__main__':
    main()