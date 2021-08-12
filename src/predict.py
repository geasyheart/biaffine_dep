# -*- coding: utf8 -*-
#

from src.biaffine_dep import BiaffineTransformerDep
from src.config import TEST_FILE
import os
dep = BiaffineTransformerDep()

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'savepoint', 'max_score.pt')


dep.predict(
    TEST_FILE,
    transformer='hfl/chinese-electra-180g-small-discriminator',
    model_path=model_path,
)
