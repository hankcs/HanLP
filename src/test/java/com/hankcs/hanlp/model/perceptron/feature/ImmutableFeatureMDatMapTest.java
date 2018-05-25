package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import junit.framework.TestCase;

public class ImmutableFeatureMDatMapTest extends TestCase
{
    public void testCompress() throws Exception
    {
        LinearModel model = new LinearModel(HanLP.Config.PerceptronCWSModelPath);
        model.compress(0.1);
    }
}