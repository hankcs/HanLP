package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class CRFPOSTaggerTest extends TestCase
{
    public static final String CORPUS = "data/test/pku98/199801.txt";
    public static String POS_MODEL_PATH = HanLP.Config.CRFPOSModelPath;
    public void testTrain() throws Exception
    {
        CRFTagger tagger = new CRFPOSTagger(null);
        tagger.train(CORPUS, POS_MODEL_PATH);
    }

    public void testLoad() throws Exception
    {
        CRFTagger tagger = new CRFPOSTagger(POS_MODEL_PATH);
    }

    public void testConvert() throws Exception
    {
        CRFTagger tagger = new CRFPOSTagger(null);
        tagger.convertCorpus(CORPUS, "data/test/crf/pos-corpus.tsv");
    }

    public void testDumpTemplate() throws Exception
    {
        CRFTagger tagger = new CRFPOSTagger(null);
        tagger.dumpTemplate("data/test/crf/pos-template.txt");
    }
}