package com.hankcs.hanlp.model.hmm;

import junit.framework.TestCase;

public class HMMSegmenterTest extends TestCase
{
    public void testTrain() throws Exception
    {
        HMMSegmenter segmenter = new HMMSegmenter();
        segmenter.train("data/test/my_cws_corpus.txt");
        System.out.println(segmenter.segment("商品和服务"));
    }
}