# -*- coding: utf8 -*-
#
import json
import os
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.config import LABEL_MAP_FILE, TRAIN_FILE, TOKENIZER_MAP_FILE


def read_conllx(file,
                field_names=('ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                             'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC')
                ):
    sent = {k: [] for k in field_names}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                yield sent
                sent = {k: [] for k in field_names}
            else:
                cells = line.split('\t')
                for key, value in zip(field_names, cells):
                    if key in ('ID', 'HEAD'):
                        value = int(value)
                    sent[key].append(value)
    if sent['ID']:
        yield sent


def get_labels(train_file: str = TRAIN_FILE):
    if os.path.exists(LABEL_MAP_FILE):
        with open(LABEL_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

    label_map = {}
    for sent in read_conllx(file=train_file):
        label_map.update({key: None for key in sent['DEPREL']})

    for index, key in enumerate(label_map.keys()):
        label_map[key] = index + 1

    label_map['PAD'] = 0
    with open(LABEL_MAP_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps(label_map, indent=2, ensure_ascii=False))

    return label_map


class DepDataSet(Dataset):
    def __init__(self, file: str, batch_size: int = 32, shuffle: bool = False,
                 device: torch.device = 'cpu'):
        self.file = file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.label_map = get_labels()

        self.data: List[Dict] = sorted([i for i in read_conllx(file=file)], key=lambda x: -len(x['FORM']))
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        if os.path.exists(TOKENIZER_MAP_FILE):
            with open(TOKENIZER_MAP_FILE, 'r', encoding='utf-8') as f:
                return json.loads(f.read())

        tokenizer = {'PAD': 0, 'UNK': 1}
        for item in self.data:
            for word in item['FORM']:
                tokenizer.setdefault(word, len(tokenizer))

        with open(TOKENIZER_MAP_FILE, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer, indent=2, ensure_ascii=False))
        return tokenizer

    def __getitem__(self, item):
        sent = self.data[item]
        arc_sent = sent['FORM']
        arc_head = sent['HEAD']
        arc_dep = sent['DEPREL']

        arc_dep_ids = [self.label_map[i] for i in arc_dep]

        sen_len = len(arc_sent)
        matrix = torch.full(size=(sen_len, sen_len), fill_value=self.label_map['pad'])

        for cur_index, target_index in enumerate(arc_head):
            if target_index == 0:
                target_index = 1
            matrix[cur_index, target_index - 1] = arc_dep_ids[cur_index]

        # 此处尝试截断，为什么呢？一是准确率一直提不上去，各种超参也不会有大的提升，一个是矩阵太稀疏
        # 另外lstm在处理超过100多个字符的时候准确率会往下滑，所以此处设置成128
        # max_seq_len = 128
        # arc_sent = arc_sent[:max_seq_len]
        # matrix = matrix[:max_seq_len, :max_seq_len]
        # 但是实验下来貌似影响不大
        #
        return torch.tensor([self.tokenizer.get(word, self.tokenizer['UNK']) for word in arc_sent],
                            dtype=torch.long), matrix

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        texts = pad_sequence([i[0] for i in batch], batch_first=True)
        matrix = [i[1] for i in batch]

        pad_matrix = torch.full(
            size=(len(matrix), max([i.size(0) for i in matrix]), max([i.size(0) for i in matrix])),
            fill_value=-100,  # 对应交叉熵的ignore_index.
            dtype=torch.long
        )
        pad_mask = torch.zeros(len(matrix), max([i.size(0) for i in matrix]), max([i.size(0) for i in matrix]),
                               dtype=torch.long)

        for index, m in enumerate(matrix):
            a, b = m.shape
            assert a == b
            pad_matrix[index][:a, :b] = m
            pad_mask[index][:a, :b] = 1
        return texts.to(self.device), pad_matrix.to(self.device), pad_mask.to(self.device)

    def to_dataloader(self, ):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)


if __name__ == '__main__':
    d = DepDataSet(file=TRAIN_FILE, batch_size=32).to_dataloader()

    for i in d:
        print(i)
