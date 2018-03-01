/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM4:36</create-date>
 *
 * <copyright file="IdLabelMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.common;

/**
 * 从id到label的映射
 * @author hankcs
 */
public interface IIdStringMap
{
    String stringOf(int id);
}
