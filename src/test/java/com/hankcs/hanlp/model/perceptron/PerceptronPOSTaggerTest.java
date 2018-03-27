package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class PerceptronPOSTaggerTest extends TestCase
{
    public void testCompress() throws Exception
    {
        PerceptronPOSTagger tagger = new PerceptronPOSTagger();
        tagger.getModel().compress(0.01);
        double[] scores = tagger.evaluate("data/test/pku98/199801.txt");
        System.out.println(scores[0]);
        tagger.getModel().save(HanLP.Config.PerceptronPOSModelPath + ".small");
    }
}