package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.datrie.MutableDoubleArrayTrieInteger;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import junit.framework.TestCase;

import java.util.Map;
import java.util.TreeMap;

public class ImmutableFeatureMDatMapTest extends TestCase
{
    public void testCompress() throws Exception
    {
        LinearModel model = new LinearModel(HanLP.Config.PerceptronCWSModelPath);
        model.compress(0.1);
    }

    public void testFeatureMap() throws Exception
    {
        LinearModel model = new LinearModel(HanLP.Config.PerceptronCWSModelPath);
        ImmutableFeatureMDatMap featureMap = (ImmutableFeatureMDatMap) model.featureMap;
        MutableDoubleArrayTrieInteger dat = featureMap.dat;
        System.out.println(featureMap.size());
        System.out.println(featureMap.entrySet().size());
        System.out.println(featureMap.idOf("\u0001/\u00014"));
        TreeMap<String, Integer> map = new TreeMap<String, Integer>();
        for (Map.Entry<String, Integer> entry : dat.entrySet())
        {
            map.put(entry.getKey(), entry.getValue());
            assertEquals(entry.getValue().intValue(), dat.get(entry.getKey()));
        }
        System.out.println(map.size());
        assertEquals(dat.size(), map.size());
    }
}