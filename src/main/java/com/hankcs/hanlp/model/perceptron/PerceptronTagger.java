/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-11-18 下午10:18</create-date>
 *
 * <copyright file="PerceptronTagger.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.model.StructuredPerceptron;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;

/**
 * 抽象的感知机标注器
 *
 * @author hankcs
 */
public abstract class PerceptronTagger
{
    /**
     * 用StructurePerceptron实现在线学习
     */
    protected final StructuredPerceptron model;

    public PerceptronTagger(LinearModel model)
    {
        assert model != null;
        this.model = model instanceof StructuredPerceptron ? (StructuredPerceptron) model : new StructuredPerceptron(model.featureMap, model.parameter);
    }

    public PerceptronTagger(StructuredPerceptron model)
    {
        assert model != null;
        this.model = model;
    }

    /**
     * 在线学习
     *
     * @param instance
     * @return
     */
    public boolean learn(Instance instance)
    {
        if (instance == null) return false;
        model.update(instance);
        return true;
    }

    /**
     * 在线学习
     *
     * @param sentence
     * @return
     */
    public abstract boolean learn(Sentence sentence);
}