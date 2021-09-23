# -*- coding: utf8 -*-
#
import json
import math
import os
import random
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.functional import F
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import MODEL_PATH, TOKENIZER_MAP_FILE
from src.metrics import Metrics
from src.model import BiaffineDepModel
from src.transform import DepDataSet, get_labels
from src.utils import logger


class BiaffineTransformerDep(object):
    def __init__(self):
        self.model: Optional[BiaffineDepModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_tokenizer_vocab(self):
        with open(TOKENIZER_MAP_FILE, 'r') as f:
            m = json.loads(f.read())
        return m

    def build_model(self, num_embeddings, n_labels: int = None):

        model = BiaffineDepModel(num_embeddings=num_embeddings, n_labels=n_labels)
        self.model = model
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        return model

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    def build_optimizer(
            self,
            warmup_steps: Union[float, int],
            num_training_steps: int,
            lr=1e-5, weight_decay=0.01,
    ):
        """
        https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L232
        :param warmup_steps:
        :param num_training_steps:
        :param lr:
        :param weight_decay:
        :return:
        """
        optimizer = Adam(self.model.parameters(), lr=lr, eps=1e-12)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
        return optimizer, scheduler

    @staticmethod
    def set_seed(seed=123321):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # ^^ safe to call this function even if cuda is not available
        torch.cuda.manual_seed_all(seed)

    def build_dataloader(self, file: str, batch_size: int = 32, shuffle: bool = False):
        return DepDataSet(file=file, batch_size=batch_size, shuffle=shuffle,
                          device=self.device).to_dataloader()

    def fit(self, train, dev, epoch: int = 5000, lr=1e-5, batch_size=32):
        self.set_seed()

        train_dataloader = self.build_dataloader(file=train, shuffle=True, batch_size=batch_size)
        dev_dataloader = self.build_dataloader(file=dev, shuffle=False, batch_size=batch_size)

        self.build_model(num_embeddings=len(self.get_tokenizer_vocab()), n_labels=len(get_labels()))

        criterion = self.build_criterion()
        optimizer, scheduler = self.build_optimizer(
            warmup_steps=0.1, num_training_steps=len(train_dataloader) * epoch,
            lr=lr
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, optimizer=optimizer, criterion=criterion,
                             scheduler=scheduler)

    def fit_loop(self, train, dev, epoch, optimizer, criterion, scheduler):
        min_loss = math.inf
        max_f1 = 0
        for _epoch in range(1, epoch + 1):
            total_loss = self.fit_dataloader(train=train, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
            if total_loss < min_loss:
                logger.info(f'Epoch {_epoch} save min loss {total_loss} model')
                min_loss = total_loss
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'min_loss.pt'))

            precision, recall, f1 = self.evaluate_dataloader(dev)
            if f1 > max_f1:
                max_f1 = f1
                logger.info(f'Epoch {_epoch} save max f1 {f1} model')
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'max_f1.pt'))
            train_p, train_r, train_f1 = self.evaluate_dataloader(train)
            logger.info(f'Epoch:{_epoch} F1:{f1},P:{precision},R:{recall}, loss value: {total_loss}, lr: {scheduler.get_last_lr()[0]:.4e}, train_f1: {train_f1}')
            scheduler.step()

    def fit_dataloader(self, train, optimizer, scheduler, criterion):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train, desc='Fit'):
            words, targets, mask = batch
            y_pred = self.model(words=words)
            loss = self.compute_loss(y_pred, targets, criterion)
            loss.backward()
            total_loss += loss.item()
            self.step(optimizer=optimizer, scheduler=scheduler)
        return total_loss

    def compute_loss(self, y_pred, y_true, criterion):
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y_true = y_true.view(-1)
        loss = criterion(y_pred, y_true)
        return loss

    def step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()
        metrics = Metrics()
        for batch in tqdm(dev, desc='Eval'):
            words, targets, mask = batch
            y_pred = self.model(words=words)
            metrics.step(y_pred=y_pred, y_true=targets, mask=mask)
        return metrics.summary()

    def save_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.module.state_dict(), save_path)

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path))
        else:
            self.model.module.load_state_dict(torch.load(save_path))

    @torch.no_grad()
    def predict(self, test,  model_path: str):
        if self.model is None:
            self.build_model(num_embeddings=len(self.get_tokenizer_vocab()), n_labels=len(get_labels()))
            self.load_weights(save_path=model_path)
            self.model.eval()

        test_dataloader = self.build_dataloader(file=test, shuffle=False, batch_size=1)

        for batch in test_dataloader:
            words, targets = batch
            y_pred = self.model(words=words).argmax(dim=-1)
            print('here')
