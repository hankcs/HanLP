/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-28 11:39</create-date>
 *
 * <copyright file="NERTrainer.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.instance.NERInstance;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;

/**
 * @author hankcs
 */
public class NERTrainer extends PerceptronTrainer
{
    @Override
    protected TagSet createTagSet()
    {
        return new NERTagSet();
    }

    @Override
    protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
    {
        return NERInstance.create(sentence, featureMap);
    }
}
