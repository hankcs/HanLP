/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/13 PM9:08</create-date>
 *
 * <copyright file="ITermFrequencyHolder.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.corpus;

import com.hankcs.hanlp.classification.collections.FrequencyMap;

/**
 * @author hankcs
 */
public interface ITermFrequencyHolder
{
    FrequencyMap<Integer> getTfMap();
}
