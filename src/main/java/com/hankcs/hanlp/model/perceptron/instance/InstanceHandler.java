/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-15 下午7:21</create-date>
 *
 * <copyright file="InstanceHandler.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.instance;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;

/**
 * @author hankcs
 */
public interface InstanceHandler
{
    boolean process(Sentence instance);
}
