# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 11:47
from hanlp.common.structure import SerializableDict
from hanlp.transform.table import TableTransform

STANFORD_SENTIMENT_TREEBANK_2_TRAIN = 'http://file.hankcs.com/corpus/SST2.zip#train.tsv'
STANFORD_SENTIMENT_TREEBANK_2_VALID = 'http://file.hankcs.com/corpus/SST2.zip#dev.tsv'
STANFORD_SENTIMENT_TREEBANK_2_TEST = 'http://file.hankcs.com/corpus/SST2.zip#test.tsv'

MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_TRAIN = 'http://file.hankcs.com/corpus/mrpc.zip#train.tsv'
MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_VALID = 'http://file.hankcs.com/corpus/mrpc.zip#dev.tsv'
MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_TEST = 'http://file.hankcs.com/corpus/mrpc.zip#test.tsv'


class StanfordSentimentTreebank2Transorm(TableTransform):
    pass


class MicrosoftResearchParaphraseCorpus(TableTransform):

    def __init__(self, config: SerializableDict = None, map_x=False, map_y=True, x_columns=(3, 4),
                 y_column=0, skip_header=True, delimiter='auto', **kwargs) -> None:
        super().__init__(config, map_x, map_y, x_columns, y_column, skip_header, delimiter, **kwargs)


def main():
    # _test_sst2()
    _test_mrpc()


def _test_sst2():
    transform = StanfordSentimentTreebank2Transorm()
    transform.fit(STANFORD_SENTIMENT_TREEBANK_2_TRAIN)
    transform.lock_vocabs()
    transform.label_vocab.summary()
    transform.build_config()
    dataset = transform.file_to_dataset(STANFORD_SENTIMENT_TREEBANK_2_TRAIN)
    for batch in dataset.take(1):
        print(batch)

def _test_mrpc():
    transform = MicrosoftResearchParaphraseCorpus()
    transform.fit(MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_VALID)
    transform.lock_vocabs()
    transform.label_vocab.summary()
    transform.build_config()
    dataset = transform.file_to_dataset(MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_VALID)
    for batch in dataset.take(1):
        print(batch)

if __name__ == '__main__':
    main()
