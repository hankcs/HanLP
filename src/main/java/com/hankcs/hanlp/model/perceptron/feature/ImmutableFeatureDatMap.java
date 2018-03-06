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

import java.util.TreeMap;

/**
 * @author hankcs
 */
public class ImmutableFeatureDatMap extends ImmutableFeatureMap
{
    DoubleArrayTrie<Integer> dat;

    public ImmutableFeatureDatMap(TreeMap<String, Integer> featureIdMap, TagSet tagSet)
    {
        super(featureIdMap.size(), tagSet);
        dat = new DoubleArrayTrie<Integer>();
        dat.build(featureIdMap);
    }

    @Override
    public int idOf(String string)
    {
        return dat.exactMatchSearch(string);
    }
}
