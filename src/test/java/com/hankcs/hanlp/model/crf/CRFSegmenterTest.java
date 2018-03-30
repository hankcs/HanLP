package com.hankcs.hanlp.model.crf;

import junit.framework.TestCase;

import java.util.List;

public class CRFSegmenterTest extends TestCase
{

    public static final String CWS_MODEL_PATH = "data/test/crf/cws.bin";

    public void testTrain() throws Exception
    {
        CRFSegmenter segmenter = new CRFSegmenter(null);
        segmenter.train("data/test/pku98/199801.txt", CWS_MODEL_PATH);
    }

    public void testLoad() throws Exception
    {
        CRFSegmenter segmenter = new CRFSegmenter(CWS_MODEL_PATH);
        List<String> wordList = segmenter.segment("商品和服务");
        System.out.println(wordList);
    }
}