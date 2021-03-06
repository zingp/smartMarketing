#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pathlib
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import torch
import config
import numpy as np
from tqdm import tqdm
from dataset import collate_fn


def evaluate(model, val_dataloader, epoch, teacher_forcing):
    """Evaluate the loss for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_dataloader (dataset.PairDataset): The evaluation data set.
        epoch (int): The epoch number.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')
    DEVICE = config.DEVICE
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            # Calculate loss.  Call model forward propagation
            loss = model(x,
                         x_len,
                         y,
                         len_oovs,
                         batch=batch,
                         num_batches=len(val_dataloader),
                         teacher_forcing=teacher_forcing)
            val_loss.append(loss.item())
    return np.mean(val_loss)
