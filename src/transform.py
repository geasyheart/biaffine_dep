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


def encoder_texts(texts: List[List[str]], tokenizer: BertTokenizerFast):
    max_word_len = 0
    texts_input_ids = []
    for text in texts:
        input_ids = tokenizer.batch_encode_plus(text, add_special_tokens=False)['input_ids']
        texts_input_ids.append(pad_sequence([torch.tensor(i) for i in input_ids], batch_first=True))

        max_len = max([len(i) for i in input_ids])
        if max_len > max_word_len:
            max_word_len = max_len

    matrix = torch.zeros(len(texts), max([len(t) for t in texts]), max_word_len, dtype=torch.long)

    for index, input_ids in enumerate(texts_input_ids):
        w, h = input_ids.shape
        matrix[index][:w, :h] = input_ids
    return matrix


class DepDataSet(Dataset):
    def __init__(self, file: str, batch_size: int = 32, shuffle: bool = False,
                 tokenizer: Union[str, AutoTokenizer] = '',
                 device: torch.device = 'cpu'):
        self.file = file
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.device = device

        self.label_map = get_labels()

        self.data: List[Dict] = [i for i in read_conllx(file=file)]

    def __getitem__(self, item):
        sent = self.data[item]
        # cls as bos
        arc_sent = ['[CLS]', *sent['FORM']]
        arc_head = [self.tokenizer.pad_token_id, *sent['HEAD']]
        arc_dep = [sent['DEPREL'][0], *sent['DEPREL']]
        arc_dep_ids = [self.label_map[i] for i in arc_dep]

        # return arc_sent, arc_head, arc_dep_ids
        sent.update({
            'arc_sent': arc_sent,
            'arc_head': arc_head,
            'arc_dep_ids': arc_dep_ids
        })
        return sent

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        arc_sents, arc_heads, arc_deps = [b['arc_sent'] for b in batch], [b['arc_head'] for b in batch], [
            b['arc_dep_ids'] for b in batch]
        text_embed = encoder_texts(arc_sents, tokenizer=self.tokenizer)
        arc_heads = pad_sequence([torch.tensor(i) for i in arc_heads], batch_first=True)
        arc_deps = pad_sequence([torch.tensor(i) for i in arc_deps], batch_first=True)
        return text_embed.to(self.device), arc_heads.to(self.device), arc_deps.to(self.device)

    def to_dataloader(self, ):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

def isprojective(sequence):
    pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True

def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices making up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)
def istree(sequence, proj=False, multiroot=False):
    r"""
    Checks if the arcs form an valid dependency tree.

    Args:
        sequence (list[int]):
            A list of head indices.
        proj (bool):
            If ``True``, requires the tree to be projective. Default: ``False``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Returns:
        ``True`` if the arcs form an valid tree, ``False`` otherwise.

    Examples:
        >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
        True
        >>> CoNLL.istree([3, 0, 0, 3], proj=True)
        False
    """

    if proj and not isprojective(sequence):
        return False
    n_roots = sum(head == 0 for head in sequence)
    if n_roots == 0:
        return False
    if not multiroot and n_roots > 1:
        return False
    if any(i == head for i, head in enumerate(sequence, 1)):
        return False
    return next(tarjan(sequence), None) is None

if __name__ == '__main__':
    d = DepDataSet(file=TRAIN_FILE, batch_size=2, tokenizer='ckiplab/albert-tiny-chinese').to_dataloader()

    for i in d:
        print(i)

    # texts = [
    #     ['我', '爱中国'],
    #     ['我', '爱', '中国']
    # ]
    # tokenizer = AutoTokenizer.from_pretrained('ckiplab/albert-tiny-chinese')
    # print(encoder_texts(texts, tokenizer))
