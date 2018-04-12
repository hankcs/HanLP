/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-28 15:53</create-date>
 *
 * <copyright file="PerceptronNERTagger.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.common.TaskType;
import com.hankcs.hanlp.model.perceptron.instance.NERInstance;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.IOException;

/**
 * 命名实体识别
 *
 * @author hankcs
 */
public class PerceptionNERecognizer extends PerceptronTagger implements NERecognizer
{
    final NERTagSet tagSet;

    public PerceptionNERecognizer(LinearModel nerModel)
    {
        super(nerModel);
        if (nerModel.tagSet().type != TaskType.NER)
        {
            throw new IllegalArgumentException(String.format("错误的模型类型: 传入的不是命名实体识别模型，而是 %s 模型", nerModel.featureMap.tagSet.type));
        }
        this.tagSet = (NERTagSet) model.tagSet();
    }

    public PerceptionNERecognizer(String nerModelPath) throws IOException
    {
        this(new LinearModel(nerModelPath));
    }

    /**
     * 加载配置文件指定的模型
     *
     * @throws IOException
     */
    public PerceptionNERecognizer() throws IOException
    {
        this(HanLP.Config.PerceptronNERModelPath);
    }

    public String[] recognize(String[] wordArray, String[] posArray)
    {
        NERInstance instance = new NERInstance(wordArray, posArray, model.featureMap);
        instance.tagArray = new int[instance.size()];
        model.viterbiDecode(instance);

        return instance.tags(tagSet);
    }

    @Override
    public NERTagSet getNERTagSet()
    {
        return tagSet;
    }

    /**
     * 在线学习
     *
     * @param segmentedTaggedNERSentence 人民日报2014格式的句子
     * @return 是否学习成功（失败的原因是参数错误）
     */
    public boolean learn(String segmentedTaggedNERSentence)
    {
        return learn(NERInstance.create(segmentedTaggedNERSentence, model.featureMap));
    }

    @Override
    protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
    {
        return NERInstance.create(sentence, featureMap);
    }
}
