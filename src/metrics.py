# -*- coding: utf8 -*-
#
from sklearn.metrics import precision_score, recall_score, f1_score

from src.transform import get_labels

ALL_LABELS = list(get_labels().values())


class Metrics(object):
    def __init__(self):
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.steps = 0

    def step(self, y_pred, y_true, mask):
        self.steps += 1
        y_pred = y_pred.argmax(-1) * mask

        y_pred = y_pred.view(-1).to('cpu')
        y_true = y_true.view(-1).to('cpu')

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)

        self.precision += precision
        self.recall += recall
        self.f1 += f1

    def summary(self):
        return self.precision / self.steps, self.recall / self.steps, self.f1 / self.steps


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)
