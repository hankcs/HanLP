/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM5:24</create-date>
 *
 * <copyright file="MutableFeatureMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class MutableFeatureMap extends FeatureMap
{
    public Map<String, Integer> featureIdMap;
    // TreeMap 5136
    // Bin 2712
    // DAT minutes
    // trie4j 3411

    public MutableFeatureMap(TagSet tagSet)
    {
        super(tagSet, true);
        featureIdMap = new TreeMap<String, Integer>();
        addTransitionFeatures(tagSet);
    }

    private void addTransitionFeatures(TagSet tagSet)
    {
        for (int i = 0; i < tagSet.size(); i++)
        {
            idOf("BL=" + tagSet.stringOf(i));
        }
        idOf("BL=_BL_");
    }

    public MutableFeatureMap(TagSet tagSet, Map<String, Integer> featureIdMap)
    {
        super(tagSet);
        this.featureIdMap = featureIdMap;
        addTransitionFeatures(tagSet);
    }

    @Override
    public Set<Map.Entry<String, Integer>> entrySet()
    {
        return featureIdMap.entrySet();
    }

    @Override
    public int idOf(String string)
    {
        Integer id = featureIdMap.get(string);
        if (id == null)
        {
            id = featureIdMap.size();
            featureIdMap.put(string, id);
        }

        return id;
    }

    public int size()
    {
        return featureIdMap.size();
    }

    public Set<String> featureSet()
    {
        return featureIdMap.keySet();
    }

    @Override
    public int[] allLabels()
    {
        return tagSet.allTags();
    }

    @Override
    public int bosTag()
    {
        return tagSet.size();
    }
}