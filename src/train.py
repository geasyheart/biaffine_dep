# -*- coding: utf8 -*-
#
from src.biaffine_dep import BiaffineTransformerDep


dep = BiaffineTransformerDep()

dep.fit(
    train='/home/yuzhang/.unlp/thirdparty/wakespace.lib.wfu.edu/bitstream/handle/10339/39379/LDC2013T21/data/tasks/dep/train.conllx',
    dev='/home/yuzhang/.unlp/thirdparty/wakespace.lib.wfu.edu/bitstream/handle/10339/39379/LDC2013T21/data/tasks/dep/dev.conllx',
    transformer='voidful/albert_chinese_tiny',

)
