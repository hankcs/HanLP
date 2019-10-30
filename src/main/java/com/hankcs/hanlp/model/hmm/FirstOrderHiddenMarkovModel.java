/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-08 5:34 PM</create-date>
 *
 * <copyright file="HiddenMarkovModel.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

/**
 * 一阶隐马尔可夫模型
 *
 * @author hankcs
 */
public class FirstOrderHiddenMarkovModel extends HiddenMarkovModel
{

    /**
     * 创建空白的隐马尔可夫模型以供训练
     */
    public FirstOrderHiddenMarkovModel()
    {
        this(null, null, null);
    }

    public FirstOrderHiddenMarkovModel(float[] start_probability, float[][] transition_probability, float[][] emission_probability)
    {
        super(start_probability, transition_probability, emission_probability);
        toLog();
    }

    @Override
    public int[][] generate(int length)
    {
        double[] pi = logToCdf(start_probability);
        double[][] A = logToCdf(transition_probability);
        double[][] B = logToCdf(emission_probability);
        int xy[][] = new int[2][length];
        xy[1][0] = drawFrom(pi); // 采样首个隐状态
        xy[0][0] = drawFrom(B[xy[1][0]]); // 根据首个隐状态采样它的显状态
        for (int t = 1; t < length; t++)
        {
            xy[1][t] = drawFrom(A[xy[1][t - 1]]);
            xy[0][t] = drawFrom(B[xy[1][t]]);
        }
        return xy;
    }

    @Override
    public float predict(int[] observation, int[] state)
    {
        final int time = observation.length; // 序列长度
        final int max_s = start_probability.length; // 状态种数

        float[] score = new float[max_s];

        // link[t][s] := 第t个时刻在当前状态是s时，前1个状态是什么
        int[][] link = new int[time][max_s];
        // 第一个时刻，使用初始概率向量乘以发射概率矩阵
        for (int cur_s = 0; cur_s < max_s; ++cur_s)
        {
            score[cur_s] = start_probability[cur_s] + emission_probability[cur_s][observation[0]];
        }

        // 第二个时刻，使用前一个时刻的概率向量乘以一阶转移矩阵乘以发射概率矩阵
        float[] pre = new float[max_s];
        for (int t = 1; t < observation.length; t++)
        {
            // swap(now, pre)
            float[] buffer = pre;
            pre = score;
            score = buffer;
            // end of swap
            for (int s = 0; s < max_s; ++s)
            {
                score[s] = Integer.MIN_VALUE;
                for (int f = 0; f < max_s; ++f)
                {
                    float p = pre[f] + transition_probability[f][s] + emission_probability[s][observation[t]];
                    if (p > score[s])
                    {
                        score[s] = p;
                        link[t][s] = f;
                    }
                }
            }
        }

        float max_score = Integer.MIN_VALUE;
        int best_s = 0;
        for (int s = 0; s < max_s; s++)
        {
            if (score[s] > max_score)
            {
                max_score = score[s];
                best_s = s;
            }
        }

        for (int t = link.length - 1; t >= 0; --t)
        {
            state[t] = best_s;
            best_s = link[t][best_s];
        }

        return max_score;
    }
}