/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-11-18 下午8:57</create-date>
 *
 * <copyright file="ImmutableFeatureMDatMap.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.collection.trie.datrie.MutableDoubleArrayTrieInteger;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.util.Map;

/**
 * 用MutableDoubleArrayTrie实现的ImmutableFeatureMap
 * @author hankcs
 */
public class ImmutableFeatureMDatMap extends ImmutableFeatureMap
{
    MutableDoubleArrayTrieInteger dat;

    public ImmutableFeatureMDatMap(MutableDoubleArrayTrieInteger dat, TagSet tagSet)
    {
        super(dat.size(), tagSet);
        this.dat = dat;
    }

    public ImmutableFeatureMDatMap(Map<String, Integer> featureIdMap, TagSet tagSet)
    {
        super(featureIdMap.size(), tagSet);
        dat = new MutableDoubleArrayTrieInteger(featureIdMap);
    }

    @Override
    public int idOf(String string)
    {
        return dat.get(string);
    }
}
