# -*- coding: utf8 -*-
#
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

TRAIN_FILE = os.path.join(DATA_PATH, 'train.conllx')
DEV_FILE = os.path.join(DATA_PATH, 'dev.conllx')
TEST_FILE = os.path.join(DATA_PATH, 'test.conllx')

LABEL_MAP_FILE = os.path.join(DATA_PATH, 'label_map.json')


MODEL_PATH = os.path.join(DATA_PATH, 'savepoint')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)