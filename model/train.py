#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
import pathlib
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config
from model import PGN
from evaluate import evaluate
from dataset import PairDataset
from dataset import collate_fn, SampleDataset
from utils import ScheduledSampler, config_info


def train(dataset, val_dataset, vocab, start_epoch=0):
    """训练、评估（验证）、存储模型.
    Args:
        dataset (dataset.PairDataset): 训练集.
        val_dataset (dataset.PairDataset): 验证集.
        vocab (vocab.Vocab): 训练集词表.
    """

    DEVICE = config.DEVICE

    model = PGN(vocab)
    model.load_model()
    model.to(DEVICE)
    if config.fine_tune:
        # 微调, 只训练attention.wc参数，其余参数固定
        print('Fine-tuning mode.')
        for name, params in model.named_parameters():
            if name != 'attention.wc.weight':
                params.requires_grad=False    
    print("loading data")
    train_data = SampleDataset(dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)

    print("initializing optimizer")

    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True, drop_last=True,
                                collate_fn=collate_fn)

    val_losses = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

    writer = SummaryWriter(config.log_path)
    # scheduled_sampler : 一个选择是否要进行teacher_forcing的工具类
    num_epochs =  len(range(start_epoch, config.epochs))
    scheduled_sampler = ScheduledSampler(num_epochs)
    if config.scheduled_sampling:
        print('scheduled_sampling mode.')
    teacher_forcing = True

    model.train()

    for epoch in range(start_epoch, config.epochs):
        print(config_info(config))
        batch_losses = []   # 存储每个batch的数据
        num_batches = len(train_dataloader)
        # 设置teaching force信号
        if config.scheduled_sampling:
            teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
            
        print('teacher_forcing = {}'.format(teacher_forcing)) 
        for batch, data in enumerate(train_dataloader): 
            x, y, x_len, y_len, oov, len_oovs = data
            assert not np.any(np.isnan(x.numpy()))
            if config.is_cuda: 
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)

            optimizer.zero_grad()      
            loss = model(x, x_len, y, len_oovs, 
                        batch=batch, 
                        num_batches=num_batches, 
                        teacher_forcing=teacher_forcing)
            batch_losses.append(loss.item())
            loss.backward()  

            # 梯度截断
            clip_grad_norm_(model.encoder.parameters(),
                            config.max_grad_norm)
            clip_grad_norm_(model.decoder.parameters(),
                            config.max_grad_norm)
            clip_grad_norm_(model.attention.parameters(),
                            config.max_grad_norm)
            optimizer.step()  

            # 每100个batch记录loss
            if (batch % 100) == 0:
                total_batch = int(batch + num_batches*epoch)
                batch_loss = np.mean(batch_losses)

                writer.add_scalar("Loss/Train",
                                loss.item(),
                                global_step=total_batch)
                writer.add_scalar("BatchLoss/Train",
                                batch_loss,
                                global_step=total_batch)

                print("Epoch {0}/{1} iter: {2} tarin_bmeanloss: {3}, bloss: {4}".format(
                    epoch, 
                    config.epochs, 
                    int(batch + num_batches*epoch), 
                    batch_loss, 
                    loss.item(),
                ))
        avg_val_loss = evaluate(model, val_dataloader, epoch, teacher_forcing)
        epoch_loss = np.mean(batch_losses)
        print('EPOCH: training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))
        # 存储更好的模型和loss
        if (avg_val_loss < val_losses):
            torch.save(model.encoder, config.encoder_save_name)
            torch.save(model.decoder, config.decoder_save_name)
            torch.save(model.attention, config.attention_save_name)
            torch.save(model.reduce_state, config.reduce_state_save_name)
            val_losses = avg_val_loss
            improve = "*"
        

        with open(config.losses_path, 'wb') as f:
            pickle.dump(val_losses, f)

        model.train()
    writer.close()


if __name__ == "__main__":
    dataset = PairDataset(config.data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    val_dataset = PairDataset(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset, val_dataset, vocab)
