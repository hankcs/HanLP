package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.tokenizer.NLPTokenizer;
import junit.framework.TestCase;

public class CWSTrainerTest extends TestCase
{

    public static final String SENTENCE = "香港特别行政区的张朝阳说商品和服务是三原县鲁桥食品厂的主营业务";

    public void testTrain() throws Exception
    {
        HanLP.Config.enableDebug();
        PerceptronTrainer trainer = new CWSTrainer();
        PerceptronTrainer.Result result = trainer.train(
                "data/test/pku98/199801.txt",
                Config.CWS_MODEL_FILE
        );
//        System.out.printf("准确率F1:%.2f\n", result.prf[2]);
        PerceptronSegmenter segmenter = new PerceptronSegmenter(result.model);
        // 也可以用
//        Segment segmenter = new AveragedPerceptronSegment(POS_MODEL_FILE);
        System.out.println(segmenter.segment("商品和服务?"));
    }

    public void testCWS() throws Exception
    {
        PerceptronSegmenter segmenter = new PerceptronSegmenter(Config.CWS_MODEL_FILE);
        segmenter.learn("下雨天 地面 积水");
        System.out.println(segmenter.segment("下雨天地面积水分外严重"));
    }

    public void testCWSandPOS() throws Exception
    {
        Segment segmenter = new PerceptronLexicalAnalyzer(Config.CWS_MODEL_FILE, Config.POS_MODEL_FILE);
        System.out.println(segmenter.seg(SENTENCE));
    }

    public void testCWSandPOSandNER() throws Exception
    {
        PerceptronLexicalAnalyzer segmenter = new PerceptronLexicalAnalyzer(Config.CWS_MODEL_FILE, Config.POS_MODEL_FILE, Config.NER_MODEL_FILE);
        Sentence sentence = segmenter.analyze(SENTENCE);
        System.out.println(sentence);
        System.out.println(segmenter.seg(SENTENCE));
        for (IWord word : sentence)
        {
            if (word instanceof CompoundWord)
                System.out.println(((CompoundWord) word).innerList);
        }
    }

    public void testCompareWithHanLP() throws Exception
    {
        System.out.println(NLPTokenizer.segment(SENTENCE));
    }
}