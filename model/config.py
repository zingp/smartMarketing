#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from typing import Optional


# General
hidden_size: Optional[int] = 512
dec_hidden_size: Optional[int] = 512
embed_size: Optional[int] = 512

# Data
max_vocab_size: Optional[int]= 20000
embed_file: Optional[str] = None   # 使用预训练词向量embeddings
source: Optional[str] = "train"    # 使用原始数据还是数据增强后的数据big_samples 
data_path: Optional[str] = '../files/{}.txt'.format(source)
val_data_path: Optional[str] = '../files/dev.txt'
test_data_path: Optional[str] = '../files/test.txt'
stop_word_file: Optional[str] = '../files/HIT_stop_words.txt'
max_src_len: Optional[int] = 300  # 不包含特殊tokens如EOS
max_tgt_len: Optional[int] = 100  # 不包含特殊tokens如EOS
truncate_src: Optional[bool] = True    # 是否截断
truncate_tgt: Optional[bool] = True
min_dec_steps: Optional[int] = 30
max_dec_steps: Optional[int] = 100   
enc_rnn_dropout: Optional[float] = 0.5
enc_attn: Optional[bool] = True
dec_attn: Optional[bool] = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0

# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1

pointer = True
epochs = 10
batch_size = 32
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1

if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:    
            model_name = 'pgn'
else:
    model_name = 'baseline'

encoder_save_name = '../saved_model/' + model_name + '/encoder.pt'
decoder_save_name = '../saved_model/' + model_name + '/decoder.pt'
attention_save_name = '../saved_model/' + model_name + '/attention.pt'
reduce_state_save_name = '../saved_model/' + model_name + '/reduce_state.pt'
losses_path = '../saved_model/' + model_name + '/val_losses.pkl'
log_path = '../runs/' + model_name

# Beam search
beam_size: Optional[int] = 3
alpha: Optional[float] = 0.2
beta: Optional[float] = 0.2
gamma: Optional[float] = 2000
