# -*- coding: utf8 -*-
#
from src.biaffine_dep import BiaffineTransformerDep
from src.config import DEV_FILE, TRAIN_FILE

dep = BiaffineTransformerDep()

dep.fit(
    train=TRAIN_FILE,
    dev=DEV_FILE,
    transformer='hfl/chinese-electra-180g-small-discriminator',
    batch_size=32,
    hidden_size=500,
    lr=1e-3,
)

