package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class CRFNERecognizerTest extends TestCase
{
    public static final String CORPUS = "data/test/pku98/199801.txt";
    public static String NER_MODEL_PATH = "data/model/crf/pku199801/ner.txt";
    public void testTrain() throws Exception
    {
        CRFTagger tagger = new CRFNERecognizer(null);
        tagger.train(CORPUS, NER_MODEL_PATH);
    }

    public void testLoad() throws Exception
    {
        CRFTagger tagger = new CRFNERecognizer(NER_MODEL_PATH);
    }

    public void testConvert() throws Exception
    {
        CRFTagger tagger = new CRFNERecognizer(null);
        tagger.convertCorpus(CORPUS, "data/test/crf/ner-corpus.tsv");
    }

    public void testDumpTemplate() throws Exception
    {
        CRFTagger tagger = new CRFNERecognizer(null);
        tagger.dumpTemplate("data/test/crf/ner-template.txt");
    }
}