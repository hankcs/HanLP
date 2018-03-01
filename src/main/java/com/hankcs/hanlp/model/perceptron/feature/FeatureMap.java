/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM5:23</create-date>
 *
 * <copyright file="FeatureMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.model.perceptron.common.IStringIdMap;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

/**
 * @author hankcs
 */
public abstract class FeatureMap implements IStringIdMap
{
    public abstract int size();

    public int[] allLabels()
    {
        return tagSet.allTags();
    }

    public int bosTag()
    {
        return tagSet.size();
    }

    public final TagSet tagSet;

    public FeatureMap(TagSet tagSet)
    {
        this.tagSet = tagSet;
    }
}