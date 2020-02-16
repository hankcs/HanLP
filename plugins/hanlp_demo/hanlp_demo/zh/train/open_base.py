# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-03 14:24

from hanlp.common.dataset import SortingSamplerBuilder
from hanlp.common.transform import NormalizeCharacter
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp.components.mtl.tasks.constituency import CRFConstituencyParsing
from hanlp.components.mtl.tasks.dep import BiaffineDependencyParsing
from hanlp.components.mtl.tasks.ner.tag_ner import TaggingNamedEntityRecognition
from hanlp.components.mtl.tasks.pos import TransformerTagging
from hanlp.components.mtl.tasks.sdp import BiaffineSemanticDependencyParsing
from hanlp.components.mtl.tasks.srl.bio_srl import SpanBIOSemanticRoleLabeling
from hanlp.components.mtl.tasks.tok.tag_tok import TaggingTokenization
from hanlp.datasets.ner.msra import MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TRAIN, MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_DEV, \
    MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TEST
from hanlp.datasets.parsing.ctb8 import CTB8_POS_TRAIN, CTB8_POS_DEV, CTB8_POS_TEST, CTB8_SD330_TEST, CTB8_SD330_DEV, \
    CTB8_SD330_TRAIN, CTB8_CWS_TRAIN, CTB8_CWS_DEV, CTB8_CWS_TEST, CTB8_BRACKET_LINE_NOEC_TRAIN, \
    CTB8_BRACKET_LINE_NOEC_DEV, CTB8_BRACKET_LINE_NOEC_TEST
from hanlp.datasets.parsing.semeval16 import SEMEVAL2016_TEXT_TRAIN_CONLLU, SEMEVAL2016_TEXT_TEST_CONLLU, \
    SEMEVAL2016_TEXT_DEV_CONLLU
from hanlp.datasets.srl.ontonotes5.chinese import ONTONOTES5_CONLL12_CHINESE_TEST, ONTONOTES5_CONLL12_CHINESE_DEV, \
    ONTONOTES5_CONLL12_CHINESE_TRAIN
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from hanlp.layers.transformers.relative_transformer import RelativeTransformerEncoder
from hanlp.utils.lang.zh.char_table import HANLP_CHAR_TABLE_JSON
from hanlp.utils.log_util import cprint
from tests import cdroot

cdroot()
tasks = {
    'tok': TaggingTokenization(
        CTB8_CWS_TRAIN,
        CTB8_CWS_DEV,
        CTB8_CWS_TEST,
        SortingSamplerBuilder(batch_size=32),
        max_seq_len=510,
        hard_constraint=True,
        char_level=True,
        tagging_scheme='BMES',
        lr=1e-3,
        transform=NormalizeCharacter(HANLP_CHAR_TABLE_JSON, 'token'),
    ),
    'pos': TransformerTagging(
        CTB8_POS_TRAIN,
        CTB8_POS_DEV,
        CTB8_POS_TEST,
        SortingSamplerBuilder(batch_size=32),
        hard_constraint=True,
        max_seq_len=510,
        char_level=True,
        dependencies='tok',
        lr=1e-3,
    ),
    'ner': TaggingNamedEntityRecognition(
        MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TRAIN,
        MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_DEV,
        MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TEST,
        SortingSamplerBuilder(batch_size=32),
        lr=1e-3,
        secondary_encoder=RelativeTransformerEncoder(768, k_as_x=True),
        dependencies='tok',
    ),
    'srl': SpanBIOSemanticRoleLabeling(
        ONTONOTES5_CONLL12_CHINESE_TRAIN,
        ONTONOTES5_CONLL12_CHINESE_DEV,
        ONTONOTES5_CONLL12_CHINESE_TEST,
        SortingSamplerBuilder(batch_size=32, batch_max_tokens=2048),
        lr=1e-3,
        crf=True,
        dependencies='tok',
    ),
    'dep': BiaffineDependencyParsing(
        CTB8_SD330_TRAIN,
        CTB8_SD330_DEV,
        CTB8_SD330_TEST,
        SortingSamplerBuilder(batch_size=32),
        lr=1e-3,
        tree=True,
        punct=True,
        dependencies='tok',
    ),
    'sdp': BiaffineSemanticDependencyParsing(
        SEMEVAL2016_TEXT_TRAIN_CONLLU,
        SEMEVAL2016_TEXT_DEV_CONLLU,
        SEMEVAL2016_TEXT_TEST_CONLLU,
        SortingSamplerBuilder(batch_size=32),
        lr=1e-3,
        apply_constraint=True,
        punct=True,
        dependencies='tok',
    ),
    'con': CRFConstituencyParsing(
        CTB8_BRACKET_LINE_NOEC_TRAIN,
        CTB8_BRACKET_LINE_NOEC_DEV,
        CTB8_BRACKET_LINE_NOEC_TEST,
        SortingSamplerBuilder(batch_size=32),
        lr=1e-3,
        dependencies='tok',
    )
}
mtl = MultiTaskLearning()
save_dir = 'data/model/mtl/open_tok_pos_ner_srl_dep_sdp_con_electra_base'
mtl.fit(
    ContextualWordEmbedding('token',
                            "hfl/chinese-electra-180g-base-discriminator",
                            average_subwords=True,
                            max_sequence_length=512,
                            word_dropout=.1),
    tasks,
    save_dir,
    30,
    lr=1e-3,
    encoder_lr=5e-5,
    grad_norm=1,
    gradient_accumulation=2,
    eval_trn=False,
)
cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
mtl.load(save_dir)
for k, v in tasks.items():
    v.trn = tasks[k].trn
    v.dev = tasks[k].dev
    v.tst = tasks[k].tst
metric, *_ = mtl.evaluate(save_dir)
for k, v in tasks.items():
    print(metric[k], end=' ')
print()
print(mtl('华纳音乐旗下的新垣结衣在12月21日于日本武道馆举办歌手出道活动'))
