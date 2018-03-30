package com.hankcs.hanlp.model.crf;

import junit.framework.TestCase;

public class CRFPOSTaggerTest extends TestCase
{
    public static String POS_MODEL_PATH = "data/test/crf/pos.bin";
    public void testTrain() throws Exception
    {
        CRFTagger tagger = new CRFPOSTagger(null);
        tagger.train("data/test/pku98/199801.txt", POS_MODEL_PATH);
    }
}