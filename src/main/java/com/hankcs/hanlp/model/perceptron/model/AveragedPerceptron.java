/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM4:45</create-date>
 *
 * <copyright file="AveragedPerceptron.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.model;

import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;

/**
 * 平均感知机算法学习的线性模型
 *
 * @author hankcs
 */
public class AveragedPerceptron extends LinearModel
{
    public AveragedPerceptron(FeatureMap featureMap, float[] parameter)
    {
        super(featureMap, parameter);
    }

    public AveragedPerceptron(FeatureMap featureMap)
    {
        super(featureMap);
    }

    /**
     * 根据答案和预测更新参数
     *
     * @param goldIndex    预测正确的特征函数（非压缩形式）
     * @param predictIndex 命中的特征函数
     */
    public void update(int[] goldIndex, int[] predictIndex, double[] total, int[] timestamp, int iter)
    {
        for (int i = 0; i < goldIndex.length; ++i)
        {
            if (goldIndex[i] == predictIndex[i])
                continue;
            else
            {
                update(goldIndex[i], 1, total, timestamp, iter);
                if (predictIndex[i] >= 0 && predictIndex[i] < parameter.length)
                    update(predictIndex[i], -1, total, timestamp, iter);
                else
                {
                    throw new IllegalArgumentException("更新参数时传入了非法的下标");
                }
            }
        }
    }

    private void update(int index, float value, double[] total, int[] timestamp, int current)
    {
        int passed = current - timestamp[index];
        total[index] += passed * parameter[index];
        parameter[index] += value;
        timestamp[index] = current;
    }

    public void average(double[] total, int[] timestamp, int current)
    {
        for (int i = 0; i < parameter.length; i++)
        {
            parameter[i] = (float) ((total[i] + (current - timestamp[i]) * parameter[i]) / current);
        }
    }
}
