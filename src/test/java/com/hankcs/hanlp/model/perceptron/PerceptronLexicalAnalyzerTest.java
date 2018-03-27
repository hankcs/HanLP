package com.hankcs.hanlp.model.perceptron;

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
}