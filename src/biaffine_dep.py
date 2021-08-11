# -*- coding: utf8 -*-
#
import math
import os
import random
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer

from src.config import MODEL_PATH
from src.metrics import Metrics
from src.model import BiaffineDepModel
from src.transform import DepDataSet, get_labels
from src.utils import logger
from torch.optim import lr_scheduler


class BiaffineTransformerDep(object):
    def __init__(self):
        self.model: Optional[BiaffineDepModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def build_model(self, transformer: str, hidden_size=500, n_labels: int = None):
        model = BiaffineDepModel(transformer=transformer, hidden_size=hidden_size, n_labels=n_labels)
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        return model

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction='none')

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
        if warmup_steps <= 1:
            warmup_steps = int(num_training_steps * warmup_steps)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
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
                          tokenizer=self.tokenizer,
                          device=self.device).to_dataloader()

    def get_transformer(self, transformer: str):
        if self.tokenizer:
            return self.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(transformer)
        self.tokenizer = tokenizer
        return tokenizer

    def fit(self, train, dev, transformer: str, epoch: int = 5000, lr=1e-5, batch_size=64, hidden_size=300):
        self.set_seed()

        self.get_transformer(transformer=transformer)
        train_dataloader = self.build_dataloader(file=train, shuffle=True, batch_size=batch_size)
        dev_dataloader = self.build_dataloader(file=dev, shuffle=False, batch_size=batch_size)

        self.build_model(transformer=transformer, n_labels=len(get_labels()) + 1, hidden_size=hidden_size)

        criterion = self.build_criterion()

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=0.2, num_training_steps=len(train_dataloader) * epoch,
            lr=lr
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, criterion=criterion, optimizer=optimizer,
                             scheduler=scheduler)

    def fit_loop(self, train, dev, epoch, criterion, optimizer, scheduler):
        min_loss = math.inf
        max_f1 = 0
        for epoch in tqdm(range(1, epoch + 1)):
            total_loss = self.fit_dataloader(train=train, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
            if total_loss < min_loss:
                logger.info(f'Epoch {epoch} save min loss {total_loss} model')
                min_loss = total_loss
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'min_loss.pt'))

            precision, recall, f1 = self.evaluate_dataloader(dev)
            if f1 > max_f1:
                max_f1 = f1
                logger.info(f'Epoch {epoch} save max f1 {f1} model')
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'max_f1.pt'))
            logger.info(f'Epoch {epoch} f1 value: {f1}, loss value: {total_loss}, lr: {scheduler.get_last_lr()[0]:.4e}')

    def fit_dataloader(self, train, criterion, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train, desc='Fit'):
            subwords, label_mask, mask = batch
            y_pred = self.model(subwords=subwords)
            loss = self.compute_loss(y_pred, label_mask, criterion, mask)
            loss.backward()
            total_loss += loss.item()
            self.step(optimizer=optimizer, scheduler=scheduler)
        return total_loss

    def compute_loss(self, y_pred, y_true, criterion, mask):
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y_true = y_true.view(-1)
        loss = criterion(y_pred, y_true)
        loss *= mask.view(-1)
        return torch.sum(loss) / loss.size(0)

    def step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()
        metrics = Metrics()
        for batch in tqdm(dev, desc='Eval'):
            subwords, label_mask, mask = batch
            y_pred = self.model(subwords=subwords)
            metrics.step(y_pred=y_pred, y_true=label_mask, mask=mask)
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
    def predict(self, test, transformer: str, model_path: str, hidden_size=300):
        self.get_transformer(transformer=transformer)
        if self.model is None:
            self.build_model(transformer=transformer, n_labels=len(get_labels()) + 1, hidden_size=hidden_size)
            self.load_weights(save_path=model_path)
            self.model.eval()

        test_dataloader = self.build_dataloader(file=test, shuffle=False, batch_size=1)

        for batch in test_dataloader:
            subwords, label_mask, mask = batch
            y_pred = self.model(subwords=subwords)
            print('here')


