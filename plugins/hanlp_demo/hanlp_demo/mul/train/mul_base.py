# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-03 14:24
from hanlp.common.dataset import SortingSamplerBuilder
from hanlp.common.transform import NormalizeToken
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp.components.mtl.tasks.tok.tag_tok import TaggingTokenization
from hanlp.components.mtl.tasks.ud import UniversalDependenciesParsing
from hanlp.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from hanlp.datasets.parsing.ud.ud210m import UD_210_MULTILINGUAL_TRAIN, UD_210_MULTILINGUAL_DEV, \
    UD_210_MULTILINGUAL_TEST
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from hanlp.utils.log_util import cprint
from tests import cdroot


def main():
    cdroot()
    transformer = "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
    tasks = {
        'tok': TaggingTokenization(
            'data/mtl/mul/tok/train.tsv',
            'data/mtl/mul/tok/dev.tsv',
            'data/mtl/mul/tok/test.tsv',
            SortingSamplerBuilder(batch_size=128, batch_max_tokens=12800),
            hard_constraint=True,
            tagging_scheme='BMES',
            delimiter='\t',
            max_seq_len=256,
            char_level=True,
            lr=1e-3,
        ),
        'ud': UniversalDependenciesParsing(
            UD_210_MULTILINGUAL_TRAIN,
            UD_210_MULTILINGUAL_DEV,
            UD_210_MULTILINGUAL_TEST,
            SortingSamplerBuilder(batch_size=128, batch_max_tokens=12800),
            lr=1e-3,
            dependencies='tok',
            max_seq_len=256,
        ),
    }
    mtl = MultiTaskLearning()
    save_dir = 'data/model/mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_mMiniLMv2L12'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    mtl.fit(
        ContextualWordEmbedding(
            'token',
            transformer,
            average_subwords=True,
            max_sequence_length=512,
            word_dropout=.2,
        ),
        tasks,
        save_dir,
        30,
        lr=1e-3,
        encoder_lr=5e-5,
        grad_norm=1,
        gradient_accumulation=8,
        eval_trn=False,
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        tau=0.5,
        cache='data/cache/ud/mtl',
    )
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    mtl.load(save_dir)
    mtl['tok'].dict_force = {"'s", "n't", "'ll", "'m", "'d", "'ve", "'re"}
    mtl['ud'].config.tree = True
    mtl.save_config(save_dir)
    for k, v in mtl.tasks.items():
        v.trn = tasks[k].trn
        v.dev = tasks[k].dev
        v.tst = tasks[k].tst
    mtl.evaluate(save_dir)
    doc = mtl(['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.',
               '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
               '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'])
    doc.pretty_print()


if __name__ == '__main__':
    main()
