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

    def step(self, y_pred, y_true):
        self.steps += 1
        #
        mask = y_true.not_equal(0)
        mask[:, 0, 0] = False

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
