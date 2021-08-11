# -*- coding: utf8 -*-
#
import json
import os
from typing import Dict, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AutoTokenizer

from src.config import LABEL_MAP_FILE, TRAIN_FILE


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
    with open(LABEL_MAP_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps(label_map, indent=2, ensure_ascii=False))

    return label_map


def encoder_texts(texts: List[List[str]], tokenizer: BertTokenizerFast, max_sequence_len=256):
    max_word_len = 0
    texts_input_ids = []
    for text in texts:
        input_ids = tokenizer.batch_encode_plus(text, add_special_tokens=False)['input_ids']
        texts_input_ids.append(pad_sequence([torch.tensor(i) for i in input_ids], batch_first=True))

        max_len = max([len(i) for i in input_ids])
        if max_len > max_word_len:
            max_word_len = max_len

    matrix = torch.zeros(len(texts), max_sequence_len, max_word_len, dtype=torch.long)

    for index, input_ids in enumerate(texts_input_ids):
        w, h = input_ids.shape
        matrix[index][:w, :h] = input_ids
    return matrix


class DepDataSet(Dataset):
    def __init__(self, file: str, batch_size: int = 32, shuffle: bool = False,
                 tokenizer: Union[str, AutoTokenizer] = '', max_len=256,
                 device: torch.device = 'cpu'):
        self.file = file
        self.batch_size = batch_size
        self.max_len = max_len

        self.shuffle = shuffle
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.device = device

        self.label_map = get_labels()

        self.data: List[Dict] = [i for i in read_conllx(file=file)]

    def __getitem__(self, item):
        sent = self.data[item]
        # cls as bos, 变成一个定长问题,如果超出256范围就会报错
        arc_sent = ['[CLS]', *sent['FORM']][:self.max_len]
        arc_head = [self.tokenizer.pad_token_id, *sent['HEAD']][:self.max_len]
        arc_dep = [sent['DEPREL'][0], *sent['DEPREL']][:self.max_len]

        arc_dep_ids = [self.label_map[i] for i in arc_dep]

        sen_len = len(arc_sent)
        matrix = torch.zeros(sen_len, sen_len)

        for cur_index, target_index in enumerate(arc_head):
            matrix[cur_index, target_index] = arc_dep_ids[cur_index]
        mask = torch.ones(sen_len, sen_len)
        mask[0, 0] = 0  # ignore root
        return arc_sent, matrix, mask

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        text_embed = encoder_texts([i[0] for i in batch], tokenizer=self.tokenizer, max_sequence_len=self.max_len)
        label_embed = [i[1] for i in batch]
        mask_embed = [i[2] for i in batch]

        final_label_embed = torch.zeros(len(batch), self.max_len, self.max_len, dtype=torch.long)
        final_mask_embed = torch.zeros(len(batch), self.max_len, self.max_len, dtype=torch.long)
        for i, label in enumerate(label_embed):
            s, e = label.shape
            final_label_embed[i][:s, :e] = label

            mask = mask_embed[i]
            assert mask.shape == (s, e)
            final_mask_embed[i][:s, :e] = mask

        # batch_size, sequence_length equal.
        assert text_embed.shape[:2] == final_label_embed.shape[:2] == final_mask_embed.shape[:2]
        return text_embed.to(self.device), final_label_embed.to(self.device), final_mask_embed.to(self.device)

    def to_dataloader(self, ):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)


if __name__ == '__main__':
    d = DepDataSet(file=TRAIN_FILE, batch_size=32, tokenizer='ckiplab/albert-tiny-chinese').to_dataloader()

    for i in d:
        print(i)

    # texts = [
    #     ['我', '爱中国'],
    #     ['我', '爱', '中国']
    # ]
    # tokenizer = AutoTokenizer.from_pretrained('ckiplab/albert-tiny-chinese')
    # print(encoder_texts(texts, tokenizer))
