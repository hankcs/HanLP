package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

public class PerceptronLexicalAnalyzerTest extends TestCase
{
    public void testLearn() throws Exception
    {
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer(Config.CWS_MODEL_FILE, Config.POS_MODEL_FILE);
        analyzer.learn("我/r 在/p 浙江/ns 金华/ns 出生/v");
        System.out.println(analyzer.analyze("我在浙江金华出生"));
        System.out.println(analyzer.analyze("我的名字叫金华"));
    }
}