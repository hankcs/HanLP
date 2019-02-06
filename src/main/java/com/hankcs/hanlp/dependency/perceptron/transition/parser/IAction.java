/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-04 上午10:23</create-date>
 *
 * <copyright file="IAction.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;

/**
 * @author hankcs
 */
public interface IAction
{
    void commit(int relation, float score, int relationSize, Configuration config);
}
