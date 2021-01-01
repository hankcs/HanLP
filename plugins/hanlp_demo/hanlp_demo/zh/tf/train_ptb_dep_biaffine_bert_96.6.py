# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-07 23:48

from hanlp.datasets.parsing.ptb import PTB_SD330_DEV, PTB_SD330_TRAIN, PTB_SD330_TEST, PTB_TOKEN_MAPPING
from hanlp.components.parsers.biaffine_parser_tf import BiaffineTransformerDependencyParserTF
from tests import cdroot
from hanlp.metrics.parsing import conllx_eval

cdroot()
save_dir = 'data/model/dep/ptb_bert_96.61'
parser = BiaffineTransformerDependencyParserTF()
# parser.fit(PTB_SD330_TRAIN, PTB_SD330_DEV, save_dir, 'bert-base-uncased',
#            batch_size=3000,
#            warmup_steps_ratio=.1,
#            token_mapping=PTB_TOKEN_MAPPING,
#            samples_per_batch=150,
#            )
parser.load(save_dir)
output = f'{save_dir}/test.predict.conll'
parser.evaluate(PTB_SD330_TEST, save_dir, warm_up=False, output=output)
uas, las = conllx_eval.evaluate(PTB_SD330_TEST, output)
print(f'Official UAS: {uas:.4f} LAS: {las:.4f}')
print(f'Model saved in {save_dir}')
