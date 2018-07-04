package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;
import junit.framework.TestCase;

import java.util.Arrays;

public class CRFPOSTaggerTest extends TestCase
{
    public static final String CORPUS = "data/test/pku98/199801.txt";
    public static String POS_MODEL_PATH = HanLP.Config.CRFPOSModelPath;

    public void testTrain() throws Exception
    {
        CRFPOSTagger tagger = new CRFPOSTagger(null); // 创建空白标注器
        tagger.train(PKU.PKU199801_TRAIN, PKU.POS_MODEL); // 训练
        System.out.println(Arrays.toString(tagger.tag("他", "的", "希望", "是", "希望", "上学"))); // 预测
        AbstractLexicalAnalyzer analyzer = new AbstractLexicalAnalyzer(new PerceptronSegmenter(), tagger); // 构造词法分析器
        System.out.println(analyzer.analyze("李狗蛋的希望是希望上学")); // 分词+词性标注
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