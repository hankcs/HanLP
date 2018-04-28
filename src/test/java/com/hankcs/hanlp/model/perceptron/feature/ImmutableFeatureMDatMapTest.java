package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import junit.framework.TestCase;

import java.util.Map;
import java.util.TreeSet;

public class ImmutableFeatureMDatMapTest extends TestCase
{
    public void testCompress() throws Exception
    {
        LinearModel model = new LinearModel(HanLP.Config.PerceptronCWSModelPath);
        TreeSet<Integer> ids = new TreeSet<Integer>();
        for (Map.Entry<String, Integer> entry : ((ImmutableFeatureMDatMap) model.featureMap).dat.entrySet())
        {
            if (entry.getValue() * model.tagSet().size() >= model.parameter.length)
            {
//                System.out.println(entry);
            }
            ids.add(entry.getValue());
        }

        System.out.println(ids.size());
        System.out.println(ids.descendingIterator().next() + 1);
        System.out.println(model.parameter.length / model.tagSet().size());
//        model.save(HanLP.Config.PerceptronCWSModelPath, 0.1);
    }
}