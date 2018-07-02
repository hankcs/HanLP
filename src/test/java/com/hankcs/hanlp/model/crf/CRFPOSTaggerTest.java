package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

import java.util.Arrays;

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
        CRFPOSTagger tagger = new CRFPOSTagger("data/model/crf/pku199801/pos.txt");
        System.out.println(Arrays.toString(tagger.tag("我", "的", "希望", "是", "希望", "和平")));
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