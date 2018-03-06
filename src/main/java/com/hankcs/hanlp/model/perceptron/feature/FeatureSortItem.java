/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-10 PM7:30</create-date>
 *
 * <copyright file="FeatureSortItem.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.feature;

import java.util.Map;

/**
 * @author hankcs
 */
public class FeatureSortItem
{
    public String key;
    public Integer id;
    public float total;

    public FeatureSortItem(Map.Entry<String, Integer> entry, float[] parameter, int tagSetSize)
    {
        key = entry.getKey();
        id = entry.getValue();
        for (int i = 0; i < tagSetSize; ++i)
        {
            total += Math.abs(parameter[id * tagSetSize + i]);
        }
    }
}
