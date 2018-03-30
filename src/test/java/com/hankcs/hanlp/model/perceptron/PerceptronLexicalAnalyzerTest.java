package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.util.List;

public class PerceptronLexicalAnalyzerTest extends TestCase
{
    PerceptronLexicalAnalyzer analyzer;

    @Override
    public void setUp() throws Exception
    {
        analyzer = new PerceptronLexicalAnalyzer(Config.CWS_MODEL_FILE, Config.POS_MODEL_FILE, Config.NER_MODEL_FILE);
    }

    public void testLearn() throws Exception
    {
        analyzer.learn("我/r 在/p 浙江/ns 金华/ns 出生/v");
        System.out.println(analyzer.analyze("我在浙江金华出生"));
        System.out.println(analyzer.analyze("我的名字叫金华"));
    }

    public void testEmptyInput() throws Exception
    {
        analyzer.segment("");
        analyzer.seg("");
    }

    public void testCustomDictionary() throws Exception
    {
        analyzer.enableCustomDictionary(true);
        assertTrue(CustomDictionary.contains("一字长蛇阵"));
        final String text = "张飞摆出一字长蛇阵如入无人之境，孙权惊呆了";
//        System.out.println(analyzer.analyze(text));
        assertTrue(analyzer.analyze(text).toString().contains(" 一字长蛇阵/"));
    }

    public void testIndexMode() throws Exception
    {
        analyzer.enableIndexMode(true);
        String text = "来到美国纽约现代艺术博物馆参观";
        List<Term> termList = analyzer.seg(text);
        assertEquals("[来到/v, 美国纽约现代艺术博物馆/ns, 美国/ns, 纽约/ns, 现代/t, 艺术/n, 博物馆/n, 参观/v]", termList.toString());
        for (Term term : termList)
        {
            assertEquals(term.word, text.substring(term.offset, term.offset + term.length()));
        }
        analyzer.enableIndexMode(false);
    }

    public void testOffset() throws Exception
    {
        analyzer.enableIndexMode(false);
        String text = "来到美国纽约现代艺术博物馆参观";
        List<Term> termList = analyzer.seg(text);
        for (Term term : termList)
        {
            assertEquals(term.word, text.substring(term.offset, term.offset + term.length()));
        }
    }

    public void testNormalization() throws Exception
    {
        analyzer.enableCustomDictionary(false);
        String text = "來到美國紐約現代藝術博物館參觀?";
        Sentence sentence = analyzer.analyze(text);
//        System.out.println(sentence);
        assertEquals("來到/v [美國/ns 紐約/ns 現代/t 藝術/n 博物館/n]/ns 參觀/v ?/w", sentence.toString());
        List<Term> termList = analyzer.seg(text);
//        System.out.println(termList);
        assertEquals("[來到/v, 美國紐約現代藝術博物館/ns, 參觀/v, ?/w]", termList.toString());
    }
}