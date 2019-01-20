/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-27 下午5:06</create-date>
 *
 * <copyright file="PerceptronPOSTagger.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.instance.POSInstance;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.common.TaskType;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.tokenizer.lexical.POSTagger;

import java.io.IOException;
import java.util.List;

/**
 * 词性标注器
 *
 * @author hankcs
 */
public class PerceptronPOSTagger extends PerceptronTagger implements POSTagger
{
    public PerceptronPOSTagger(LinearModel model)
    {
        super(model);
        if (model.featureMap.tagSet.type != TaskType.POS)
        {
            throw new IllegalArgumentException(String.format("错误的模型类型: 传入的不是词性标注模型，而是 %s 模型", model.featureMap.tagSet.type));
        }
    }

    public PerceptronPOSTagger(String modelPath) throws IOException
    {
        this(new LinearModel(modelPath));
    }

    /**
     * 加载配置文件指定的模型
     *
     * @throws IOException
     */
    public PerceptronPOSTagger() throws IOException
    {
        this(HanLP.Config.PerceptronPOSModelPath);
    }

    /**
     * 标注
     *
     * @param words
     * @return
     */
    @Override
    public String[] tag(String... words)
    {
        POSInstance instance = new POSInstance(words, model.featureMap);
        return tag(instance);
    }

    public String[] tag(POSInstance instance)
    {
        instance.tagArray = new int[instance.featureMatrix.length];

        model.viterbiDecode(instance, instance.tagArray);
        return instance.tags(model.tagSet());
    }

    /**
     * 标注
     *
     * @param wordList
     * @return
     */
    @Override
    public String[] tag(List<String> wordList)
    {
        String[] termArray = new String[wordList.size()];
        wordList.toArray(termArray);
        return tag(termArray);
    }

    /**
     * 在线学习
     *
     * @param segmentedTaggedSentence 人民日报2014格式的句子
     * @return 是否学习成功（失败的原因是参数错误）
     */
    public boolean learn(String segmentedTaggedSentence)
    {
        return learn(POSInstance.create(segmentedTaggedSentence, model.featureMap));
    }

    /**
     * 在线学习
     *
     * @param wordTags [单词]/[词性]数组
     * @return 是否学习成功（失败的原因是参数错误）
     */
    public boolean learn(String... wordTags)
    {
        String[] words = new String[wordTags.length];
        String[] tags = new String[wordTags.length];
        for (int i = 0; i < wordTags.length; i++)
        {
            String[] wordTag = wordTags[i].split("//");
            words[i] = wordTag[0];
            tags[i] = wordTag[1];
        }
        return learn(new POSInstance(words, tags, model.featureMap));
    }

    @Override
    protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
    {
        for (Word word : sentence.toSimpleWordList())
        {
            if (!model.featureMap.tagSet.contains(word.getLabel()))
                throw new IllegalArgumentException("在线学习不可能学习新的标签: " + word + " ；请标注语料库后重新全量训练。");
        }
        return POSInstance.create(sentence, featureMap);
    }
}
