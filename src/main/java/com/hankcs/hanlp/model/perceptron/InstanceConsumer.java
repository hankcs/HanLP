/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-15 下午7:34</create-date>
 *
 * <copyright file="InstanceConsumer.java" company="码农场">
 * Copyright (c) 2018, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.utility.IOUtility;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.utility.Utility;

import java.io.IOException;

/**
 * 需要处理实例的消费者
 *
 * @author hankcs
 */
public abstract class InstanceConsumer
{
    private static char[] tableChar;

    static
    {
        tableChar = new char[CharTable.CONVERT.length];
        System.arraycopy(CharTable.CONVERT, 0, tableChar, 0, tableChar.length);
        for (int c = 0; c <= 32; ++c)
        {
            tableChar[c] = '&'; // 也可以考虑用 '。'
        }
    }

    protected abstract Instance createInstance(Sentence sentence, final FeatureMap featureMap);

    protected double[] evaluate(String developFile, String modelFile) throws IOException
    {
        return evaluate(developFile, new LinearModel(modelFile));
    }

    protected double[] evaluate(String developFile, final LinearModel model) throws IOException
    {
        final int[] stat = new int[2];
        IOUtility.loadInstance(developFile, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                Utility.normalize(sentence);
                Instance instance = createInstance(sentence, model.featureMap);
                IOUtility.evaluate(instance, model, stat);
                return false;
            }
        });

        return new double[]{stat[1] / (double) stat[0] * 100};
    }

    protected String normalize(String text)
    {
        char[] result = new char[text.length()];
        for (int i = 0; i < result.length; i++)
        {
            result[i] = tableChar[text.charAt(i)];
        }
        return new String(result);
    }
}
