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


def train(dataset, val_dataset, vocab):
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
        # In fine-tuning mode, 只训练attention.wc参数，其余参数固定
        print('Fine-tuning mode.')
        for name, params in model.named_parameters():
            if name != 'attention.wc.weight':
                params.requires_grad=False    
    # forward
    print("loading data")
    train_data = SampleDataset(dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)

    print("initializing optimizer")

    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_losses = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

    # torch.cuda.empty_cache()
    # SummaryWriter: Log writer used for TensorboardX visualization.
    writer = SummaryWriter(config.log_path)
    # tqdm: A tool for drawing progress bars during training.
    # scheduled_sampler : A tool for choosing teacher_forcing or not
    teacher_forcing = False
    if config.scheduled_sampling:
        print('scheduled_sampling mode.')
        teacher_forcing = True

    
    for epoch in range(config.epochs):
        print(config_info(config))
        batch_losses = []  # Get loss of each batch.
        num_batches = len(train_dataloader)
        # set a teacher_forcing signal
        print('teacher_forcing = {}'.format(teacher_forcing))
        model.train()  # Sets the module in training mode.
        #with tqdm(total=num_batches//100) as batch_progress:
        for batch, data in enumerate(train_dataloader):
            x, y, x_len, y_len, oov, len_oovs = data
            assert not np.any(np.isnan(x.numpy()))
            if config.is_cuda:  # Training with GPUs.
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)

            optimizer.zero_grad()  # Clear gradients.
            # Calculate loss.  Call model forward propagation
            loss = model(x, x_len, y, len_oovs, 
                        batch=batch, 
                        num_batches=num_batches, 
                        teacher_forcing=teacher_forcing)
            batch_losses.append(loss.item())
            loss.backward()  # Backpropagation.

            # Do gradient clipping to prevent gradient explosion.
            clip_grad_norm_(model.encoder.parameters(),
                            config.max_grad_norm)
            clip_grad_norm_(model.decoder.parameters(),
                            config.max_grad_norm)
            clip_grad_norm_(model.attention.parameters(),
                            config.max_grad_norm)
            optimizer.step()  # Update weights.

            # Output and record epoch loss every 100 batches.
            if (batch % 100) == 0:
                # batch_progress.set_description(f'Epoch {epoch}')
                # batch_progress.set_postfix(Batch=batch,
                #                             Loss=loss.item())
                # batch_progress.update()
                # Write loss for tensorboard.
                print("Epoch {}/{} batch: {} loss: {}".format(epoch,
                            config.epochs, batch, np.mean(batch_losses)))
                writer.add_scalar(f'Average loss for epoch {epoch}',
                                    np.mean(batch_losses),
                                    global_step=batch)
        # Calculate average loss over all batches in an epoch.
        epoch_loss = np.mean(batch_losses)

        # epoch_progress.set_description(f'Epoch {epoch}')
        # epoch_progress.set_postfix(Loss=epoch_loss)
        # epoch_progress.update()

        avg_val_loss = evaluate(model, val_data, epoch, teacher_forcing)

        print('training loss:{}'.format(epoch_loss),
                'validation loss:{}'.format(avg_val_loss))

        # Update minimum evaluating loss.
        if (avg_val_loss < val_losses):
            torch.save(model.encoder, config.encoder_save_name)
            torch.save(model.decoder, config.decoder_save_name)
            torch.save(model.attention, config.attention_save_name)
            torch.save(model.reduce_state, config.reduce_state_save_name)
            val_losses = avg_val_loss
        with open(config.losses_path, 'wb') as f:
            pickle.dump(val_losses, f)

    writer.close()


if __name__ == "__main__":
    # Prepare dataset for training.
    #DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
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
