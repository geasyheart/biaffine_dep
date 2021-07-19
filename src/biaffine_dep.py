# -*- coding: utf8 -*-
#
import math
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer

from src.config import MODEL_PATH
from src.metrics import Metrics
from src.model import BiaffineDepModel
from src.transform import DepDataSet, get_labels
from src.utils import logger


class BiaffineTransformerDep(object):
    def __init__(self):
        self.model: Optional[BiaffineDepModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def build_model(self, transformer: str, hidden_size=500, n_labels: int = None):
        model = BiaffineDepModel(transformer=transformer, hidden_size=hidden_size, n_labels=n_labels)
        self.model = model
        self.model.to(self.device)
        return model

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction='none')

    def build_optimizer(
            self, transformer_lr, transformer_weight_decay,
            num_warmup_steps, num_training_steps,
            pretrained: torch.nn.Module,
            lr=1e-5, weight_decay=0.01,
            no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),

    ):
        if transformer_lr is None:
            transformer_lr = lr
        if transformer_weight_decay is None:
            transformer_weight_decay = weight_decay
        params = defaultdict(lambda: defaultdict(list))
        pretrained = set(pretrained.parameters())
        if isinstance(no_decay, tuple):
            def no_decay_fn(name):
                return any(nd in name for nd in no_decay)
        else:
            assert callable(no_decay), 'no_decay has to be callable or a tuple of str'
            no_decay_fn = no_decay
        for n, p in self.model.named_parameters():
            is_pretrained = 'pretrained' if p in pretrained else 'non_pretrained'
            is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
            params[is_pretrained][is_no_decay].append(p)

        grouped_parameters = [
            {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
            {'params': params['pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': transformer_lr},
            {'params': params['non_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
            {'params': params['non_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
        ]

        optimizer = AdamW(grouped_parameters, lr=lr, weight_decay=weight_decay, eps=1e-8)

        if num_warmup_steps < 1:
            num_warmup_steps = num_warmup_steps * num_training_steps

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
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
        tokenizer = BertTokenizer.from_pretrained(transformer)
        self.tokenizer = tokenizer
        return tokenizer

    def fit(self, train, dev, transformer: str, epoch: int = 1000, lr=1e-5):
        self.set_seed()

        self.get_transformer(transformer=transformer)
        train_dataloader = self.build_dataloader(file=train, shuffle=True)
        dev_dataloader = self.build_dataloader(file=dev, shuffle=False)

        model = self.build_model(transformer=transformer, n_labels=len(get_labels()) + 1)

        criterion = self.build_criterion()

        optimizer, scheduler = self.build_optimizer(
            transformer_lr=lr, transformer_weight_decay=None, num_warmup_steps=0.1,
            num_training_steps=len(train_dataloader) * epoch, pretrained=model.encoder
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

    def fit_dataloader(self, train, criterion, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train, desc='Fit'):
            subwords, label_mask = batch
            y_pred = self.model(subwords=subwords)
            loss = self.compute_loss(y_pred, label_mask, criterion)
            loss.backward()
            total_loss += loss.item()
            self.step(optimizer=optimizer, scheduler=scheduler)
        return total_loss

    def compute_loss(self, y_pred, y_true, criterion):
        mask = y_true.not_equal(0)
        mask[:, 0, 0] = False

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
            subwords, label_mask = batch
            y_pred = self.model(subwords=subwords)
            metrics.step(y_pred=y_pred, y_true=label_mask)
        return metrics.summary()

    def save_weights(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_weights(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
