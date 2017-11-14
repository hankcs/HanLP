/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/16 AM10:37</create-date>
 *
 * <copyright file="DfFeatureData.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.features;

import com.hankcs.hanlp.classification.corpus.IDataSet;

/**
 * 包含倒排文档频次的特征数据
 * @author hankcs
 */
public class DfFeatureData extends BaseFeatureData
{
    public int[] df;
    /**
     * 构造一个空白的统计对象
     *
     * @param dataSet
     */
    public DfFeatureData(IDataSet dataSet)
    {
        super(dataSet);
    }
}
