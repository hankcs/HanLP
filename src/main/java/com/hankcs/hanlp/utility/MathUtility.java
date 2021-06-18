/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/10 PM6:51</create-date>
 *
 * <copyright file="MathUtility.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.seg.common.Vertex;

import java.util.Map;
import java.util.Set;

import static com.hankcs.hanlp.utility.Predefine.TOTAL_FREQUENCY;
import static com.hankcs.hanlp.utility.Predefine.lambda;
import static com.hankcs.hanlp.utility.Predefine.myu;

/**
 * 一些数学小工具
 *
 * @author hankcs
 */
public class MathUtility
{
    public static int sum(int... var)
    {
        int sum = 0;
        for (int x : var)
        {
            sum += x;
        }

        return sum;
    }

    public static float sum(float... var)
    {
        float sum = 0;
        for (float x : var)
        {
            sum += x;
        }

        return sum;
    }

    public static double percentage(double current, double total)
    {
        return current / total * 100.;
    }

    public static double average(double array[])
    {
        double sum = 0;
        for (int i = 0; i < array.length; i++)
            sum += array[i];
        return sum / array.length;
    }

    /**
     * 使用log-sum-exp技巧来归一化一组对数值
     *
     * @param predictionScores
     */
    public static void normalizeExp(Map<String, Double> predictionScores)
    {
        Set<Map.Entry<String, Double>> entrySet = predictionScores.entrySet();
        double max = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> entry : entrySet)
        {
            max = Math.max(max, entry.getValue());
        }

        double sum = 0.0;
        //通过减去最大值防止浮点数溢出
        for (Map.Entry<String, Double> entry : entrySet)
        {
            Double value = Math.exp(entry.getValue() - max);
            entry.setValue(value);

            sum += value;
        }

        if (sum != 0.0)
        {
            for (Map.Entry<String, Double> entry : entrySet)
            {
                predictionScores.put(entry.getKey(), entry.getValue() / sum);
            }
        }
    }

    public static void normalizeExp(double[] predictionScores)
    {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : predictionScores)
        {
            max = Math.max(max, value);
        }

        double sum = 0.0;
        //通过减去最大值防止浮点数溢出
        for (int i = 0; i < predictionScores.length; i++)
        {
            predictionScores[i] = Math.exp(predictionScores[i] - max);
            sum += predictionScores[i];
        }

        if (sum != 0.0)
        {
            for (int i = 0; i < predictionScores.length; i++)
            {
                predictionScores[i] /= sum;
            }
        }
    }

    /**
     * 从一个词到另一个词的词的花费
     *
     * @param from 前面的词
     * @param to   后面的词
     * @return 分数
     */
    public static double calculateWeight(Vertex from, Vertex to)
    {
        int fFrom = from.getAttribute().totalFrequency;
        int fBigram = CoreBiGramTableDictionary.getBiFrequency(from.wordID, to.wordID);
        int fTo = to.getAttribute().totalFrequency;
        //        logger.info(String.format("%5s frequency:%6d, %s fBigram:%3d, weight:%.2f", from.word, frequency, from.word + "@" + to.word, fBigram, value));
        return -Math.log(lambda * (myu * fBigram / (fFrom + 1) + 1 - myu) + (1 - lambda) * fTo / TOTAL_FREQUENCY);
    }

}