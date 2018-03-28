package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;

public class PerceptronLexicalAnalyzerTest extends TestCase
{
    PerceptronLexicalAnalyzer analyzer;

    @Override
    public void setUp() throws Exception
    {
        analyzer = new PerceptronLexicalAnalyzer(Config.CWS_MODEL_FILE, Config.POS_MODEL_FILE);
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
}