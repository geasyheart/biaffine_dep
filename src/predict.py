# -*- coding: utf8 -*-
#

from src.biaffine_dep import BiaffineTransformerDep
from src.config import TEST_FILE
import os
dep = BiaffineTransformerDep()

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'savepoint', 'max_f1.pt')


dep.predict(
    TEST_FILE,
    transformer='voidful/albert_chinese_tiny',
    model_path=model_path
)