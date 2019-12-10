package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.corpus.PKU;
import junit.framework.TestCase;

public class HMMLexicalAnalyzerTest extends TestCase
{

    public static final String CORPUS_PATH = PKU.PKU199801_TRAIN;

    public void testTrain() throws Exception
    {
        HMMSegmenter segmenter = new HMMSegmenter();
        segmenter.train(CORPUS_PATH);
        HMMPOSTagger tagger = new HMMPOSTagger();
        tagger.train(CORPUS_PATH);
        HMMNERecognizer recognizer = new HMMNERecognizer();
        recognizer.train(CORPUS_PATH);
        HMMLexicalAnalyzer analyzer = new HMMLexicalAnalyzer(segmenter, tagger, recognizer);
        System.out.println(analyzer.analyze("我的希望是希望人们幸福"));
    }
}