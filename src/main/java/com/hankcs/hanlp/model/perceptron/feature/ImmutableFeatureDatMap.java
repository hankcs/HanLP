/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-05 PM8:19</create-date>
 *
 * <copyright file="ImmutableFeatureMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class ImmutableFeatureDatMap extends FeatureMap
{
    DoubleArrayTrie<Integer> dat;

    public ImmutableFeatureDatMap(TreeMap<String, Integer> featureIdMap, TagSet tagSet)
    {
        super(tagSet);
        dat = new DoubleArrayTrie<Integer>();
        dat.build(featureIdMap);
    }

    @Override
    public int idOf(String string)
    {
        return dat.exactMatchSearch(string);
    }

    @Override
    public int size()
    {
        return dat.size();
    }

    @Override
    public Set<Map.Entry<String, Integer>> entrySet()
    {
        throw new UnsupportedOperationException("这份DAT实现不支持遍历");
    }
}
