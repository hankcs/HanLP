# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 11:47
from hanlp.common.dataset import TableDataset

STANFORD_SENTIMENT_TREEBANK_2_TRAIN = 'http://file.hankcs.com/corpus/SST2.zip#train.tsv'
STANFORD_SENTIMENT_TREEBANK_2_DEV = 'http://file.hankcs.com/corpus/SST2.zip#dev.tsv'
STANFORD_SENTIMENT_TREEBANK_2_TEST = 'http://file.hankcs.com/corpus/SST2.zip#test.tsv'

MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_TRAIN = 'http://file.hankcs.com/corpus/mrpc.zip#train.tsv'
MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_DEV = 'http://file.hankcs.com/corpus/mrpc.zip#dev.tsv'
MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_TEST = 'http://file.hankcs.com/corpus/mrpc.zip#test.tsv'


class SST2Dataset(TableDataset):
    pass


def main():
    dataset = SST2Dataset(STANFORD_SENTIMENT_TREEBANK_2_TEST)
    print(dataset)


if __name__ == '__main__':
    main()
