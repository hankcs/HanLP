# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-07 23:48
from hanlp.metrics.parsing import conllx_eval

from hanlp.datasets.parsing.ptb import PTB_SD330_DEV, PTB_SD330_TRAIN, PTB_SD330_TEST, PTB_TOKEN_MAPPING
from hanlp.components.parsers.biaffine_parser_tf import BiaffineTransformerDependencyParserTF
from tests import cdroot

cdroot()
save_dir = 'data/model/dep/ptb_bert_positional_diff_lr'
parser = BiaffineTransformerDependencyParserTF()
parser.fit(PTB_SD330_TRAIN, PTB_SD330_DEV, save_dir, 'bert-base-uncased',
           batch_size=3000,
           warmup_steps_ratio=.1,
           token_mapping=PTB_TOKEN_MAPPING,
           samples_per_batch=150,
           transformer_dropout=.33,
           learning_rate=1e-4,
           learning_rate_transformer=1e-5,
           d_positional=128,
           # early_stopping_patience=10,
           )
# parser.load(save_dir)
# output = f'{save_dir}/test.predict.conll'
parser.evaluate(PTB_SD330_TEST, save_dir, warm_up=False)
# uas, las = conllx_eval.evaluate(PTB_SD330_TEST, output)
# print(f'Official UAS: {uas:.4f} LAS: {las:.4f}')
# print(f'Model saved in {save_dir}')
