/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-05 PM8:39</create-date>
 *
 * <copyright file="ImmutableFeatureHashMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * @author hankcs
 */
public class ImmutableFeatureMap extends FeatureMap
{
    public Map<String, Integer> featureIdMap;

    public ImmutableFeatureMap(Map<String, Integer> featureIdMap, TagSet tagSet)
    {
        super(tagSet);
        this.featureIdMap = featureIdMap;
    }

    public ImmutableFeatureMap(Set<Map.Entry<String, Integer>> entrySet, TagSet tagSet)
    {
        super(tagSet);
        this.featureIdMap = new HashMap<String, Integer>();
        for (Map.Entry<String, Integer> entry : entrySet)
        {
            featureIdMap.put(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public int idOf(String string)
    {
        Integer id = featureIdMap.get(string);
        if (id == null) return -1;
        return id;
    }

    @Override
    public int size()
    {
        return featureIdMap.size();
    }

    @Override
    public Set<Map.Entry<String, Integer>> entrySet()
    {
        return featureIdMap.entrySet();
    }
}