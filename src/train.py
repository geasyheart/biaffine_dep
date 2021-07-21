# -*- coding: utf8 -*-
#
from src.biaffine_dep import BiaffineTransformerDep
from src.config import DEV_FILE, TRAIN_FILE

dep = BiaffineTransformerDep()

dep.fit(
    train=TRAIN_FILE,
    dev=DEV_FILE,
    transformer='voidful/albert_chinese_tiny',

)
