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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.alg import eisner, mst
from src.config import MODEL_PATH
from src.metrics import Metrics, AttachmentMetric
from src.model import BiaffineDepModel
from src.transform import DepDataSet, istree, get_labels
from src.utils import logger


class BiaffineTransformerDep(object):
    def __init__(self):
        self.model: Optional[BiaffineDepModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def build_model(self, transformer: str, n_out: int):
        model = BiaffineDepModel(transformer=transformer, n_out=n_out)
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()

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

    def fit(
            self, train, dev, transformer: str, epoch: int = 5000,
            lr=1e-5, batch_size=64,
            warmup_steps=0.1,
    ):
        self.set_seed()

        self.get_transformer(transformer=transformer)
        train_dataloader = self.build_dataloader(file=train, shuffle=True, batch_size=batch_size)
        dev_dataloader = self.build_dataloader(file=dev, shuffle=False, batch_size=batch_size)

        self.build_model(transformer=transformer, n_out=len(get_labels()) + 1)

        criterion = self.build_criterion()

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epoch,
            lr=lr
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, criterion=criterion, optimizer=optimizer,
                             scheduler=scheduler)

    def fit_loop(self, train, dev, epoch, criterion, optimizer, scheduler):
        min_loss = math.inf
        max_metric = 0
        for _epoch in range(1, epoch + 1):
            total_loss, metric = self.fit_dataloader(
                train=train, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler
            )
            if total_loss < min_loss:
                logger.info(f'Epoch {_epoch} save min loss {total_loss} model')
                min_loss = total_loss
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'min_loss.pt'))

            metric = self.evaluate_dataloader(dev)
            if metric > max_metric:
                max_metric = metric
                self.save_weights(save_path=os.path.join(MODEL_PATH, 'max_score.pt'))
            logger.info(f'Epoch {_epoch} {metric}, loss value: {total_loss}, lr: {scheduler.get_last_lr()[0]:.4e}')

    def fit_dataloader(self, train, criterion, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        metric = AttachmentMetric()
        for batch in tqdm(train, desc='Fit'):
            subwords, arcs, rels = batch
            mask = subwords.ne(self.tokenizer.pad_token_id).any(-1)
            mask[:, 0] = 0  # 忽略bos

            s_arc, s_rel = self.model(subwords=subwords)
            loss = self.compute_loss(s_arc, s_rel, arcs, rels, mask, criterion)
            loss.backward()
            total_loss += loss.item()
            self.step(optimizer=optimizer, scheduler=scheduler)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            metric(arc_preds, rel_preds, arcs, rels, mask)
        return total_loss, metric

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not istree(seq[1:i + 1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def compute_loss(self, s_arc, s_rel, arcs, rels, mask, criterion):
        """
        方便理解：

        假设
        s_arc.shape = (2, 10, 10)
        s_rel.shape = (2, 10, 10, 47)

        s_arc[mask], s_rel[mask]后为：
        s_arc.shape = (14, 10)
        s_rel.shape = (14, 10, 47)

        # 第一点
        计算s_arc loss，就为获取当前词和句子其他词中权重最大的一个，表示最可能的依存，即：
        s_arc.argmax(-1)
        由于依存的种类为句子的长度，所以此处是一个变长的分类问题，即s_arc.size(-1)是不固定的

        这一点和其他的分类稍微有点不同（不同点在于为降维到num_labels,比如768 -> 5个分类，这个5在任务开始时就已经固定了）

        # 第二点
        s_rel第二维(即：10)，表示当前词和句子中指定的index存在依存（即s_rel[torch.arange(len(arcs))], arcs），
        获取到后，就变成了一个传统的分类问题，表示当前词和指定词有47种依存关系，要获取最大的那个最为最终结果。

        :param s_arc:
        :param s_rel:
        :param arcs:
        :param rels:
        :param mask:
        :param criterion:
        :return:
        """
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]  # == s_rel[:, arcs] 表示取第二维的第几行
        arc_loss = criterion(s_arc, arcs)  # 在42 * 43 的可能性
        rel_loss = criterion(s_rel, rels)  # 在 42 * 47的可能性
        return arc_loss + rel_loss

    #
    def step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()
        metric = AttachmentMetric()
        for batch in tqdm(dev, desc='Eval'):
            subwords, arcs, rels = batch
            mask = subwords.ne(self.tokenizer.pad_token_id).any(-1)
            mask[:, 0] = 0  # 忽略bos

            s_arc, s_rel = self.model(subwords=subwords)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            metric(arc_preds, rel_preds, arcs, rels, mask)
        return metric

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
    def predict(self, test, transformer: str, model_path: str):
        self.get_transformer(transformer=transformer)
        if self.model is None:
            self.build_model(transformer=transformer, n_out=len(get_labels()) + 1)
            self.load_weights(save_path=model_path)
            self.model.eval()
            self.id_label_map = {v: k for k, v in get_labels().items()}

        test_dataloader = self.build_dataloader(file=test, shuffle=False, batch_size=2)

        preds = {'arcs': [], 'rels': []}
        # criterion = nn.CrossEntropyLoss()
        # metric = AttachmentMetric()
        # total_loss = 0
        for batch in tqdm(test_dataloader):
            subwords, arcs, rels = batch
            mask = subwords.ne(self.tokenizer.pad_token_id).any(-1)
            mask[:, 0] = 0  # 忽略bos
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(subwords=subwords)
            # loss = self.compute_loss(s_arc, s_rel, arcs, rels, mask, criterion)
            # total_loss += loss.item()
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            # metric(arc_preds, rel_preds, arcs, rels, mask)

            preds['arcs'].extend(arc_preds[mask].split(lens))
            preds['rels'].extend(rel_preds[mask].split(lens))
        # print(metric)
        # print(total_loss / len(test_dataloader))
        preds['arcs'] = [seq.tolist() for seq in preds['arcs']]
        preds['rels'] = [[self.id_label_map[i] for i in seq.tolist()] for seq in preds['rels']]
        return preds
