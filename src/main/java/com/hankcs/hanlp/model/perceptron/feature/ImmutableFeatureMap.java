/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-05 PM8:45</create-date>
 *
 * <copyright file="ImmutableFeatureMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

/**
 * @author hankcs
 */
public abstract class ImmutableFeatureMap extends FeatureMap
{
    public ImmutableFeatureMap()
    {
    }

    public ImmutableFeatureMap(TagSet tagSet)
    {
        super(tagSet);
    }
}
