/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-09 7:47 PM</create-date>
 *
 * <copyright file="HiddenMarkovModel.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.utility.MathUtility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * @author hankcs
 */
public abstract class HiddenMarkovModel
{
    /**
     * 初始状态概率向量
     */
    public float[] start_probability;
    /**
     * 观测概率矩阵
     */
    public float[][] emission_probability;
    /**
     * 状态转移概率矩阵
     */
    public float[][] transition_probability;

    /**
     * 构造隐马模型
     *
     * @param start_probability      初始状态概率向量
     * @param transition_probability 状态转移概率矩阵
     * @param emission_probability   观测概率矩阵
     */
    public HiddenMarkovModel(float[] start_probability, float[][] transition_probability, float[][] emission_probability)
    {
        this.start_probability = start_probability;
        this.transition_probability = transition_probability;
        this.emission_probability = emission_probability;
    }

    /**
     * 对数概率转为累积分布函数
     *
     * @param log
     * @return
     */
    protected static double[] logToCdf(float[] log)
    {
        double[] cdf = new double[log.length];
        cdf[0] = Math.exp(log[0]);
        for (int i = 1; i < cdf.length - 1; i++)
        {
            cdf[i] = cdf[i - 1] + Math.exp(log[i]);
        }
        cdf[cdf.length - 1] = 1.0;
        return cdf;
    }

    /**
     * 对数概率转化为累积分布函数
     *
     * @param log
     * @return
     */
    protected static double[][] logToCdf(float[][] log)
    {
        double[][] cdf = new double[log.length][log[0].length];
        for (int i = 0; i < log.length; i++)
            cdf[i] = logToCdf(log[i]);
        return cdf;
    }

    /**
     * 采样
     *
     * @param cdf 累积分布函数
     * @return
     */
    protected static int drawFrom(double[] cdf)
    {
        return -Arrays.binarySearch(cdf, Math.random()) - 1;
    }

    /**
     * 频次向量归一化为概率分布
     *
     * @param freq
     */
    protected void normalize(float[] freq)
    {
        float sum = MathUtility.sum(freq);
        for (int i = 0; i < freq.length; i++)
            freq[i] /= sum;
    }

    public void unLog()
    {
        for (int i = 0; i < start_probability.length; i++)
        {
            start_probability[i] = (float) Math.exp(start_probability[i]);
        }
        for (int i = 0; i < emission_probability.length; i++)
        {
            for (int j = 0; j < emission_probability[i].length; j++)
            {
                emission_probability[i][j] = (float) Math.exp(emission_probability[i][j]);
            }
        }
        for (int i = 0; i < transition_probability.length; i++)
        {
            for (int j = 0; j < transition_probability[i].length; j++)
            {
                transition_probability[i][j] = (float) Math.exp(transition_probability[i][j]);
            }
        }
    }

    protected void toLog()
    {
        if (start_probability == null || transition_probability == null || emission_probability == null) return;
        for (int i = 0; i < start_probability.length; i++)
        {
            start_probability[i] = (float) Math.log(start_probability[i]);
            for (int j = 0; j < start_probability.length; j++)
                transition_probability[i][j] = (float) Math.log(transition_probability[i][j]);
            for (int j = 0; j < emission_probability[0].length; j++)
                emission_probability[i][j] = (float) Math.log(emission_probability[i][j]);
        }
    }

    /**
     * 训练
     *
     * @param samples 数据集 int[i][j] i=0为观测，i=1为状态，j为时序轴
     */
    public void train(Collection<int[][]> samples)
    {
        if (samples.isEmpty()) return;
        int max_state = 0;
        int max_obser = 0;
        for (int[][] sample : samples)
        {
            if (sample.length != 2 || sample[0].length != sample[1].length) throw new IllegalArgumentException("非法样本");
            for (int o : sample[0])
                max_obser = Math.max(max_obser, o);
            for (int s : sample[1])
                max_state = Math.max(max_state, s);
        }
        estimateStartProbability(samples, max_state);
        estimateTransitionProbability(samples, max_state);
        estimateEmissionProbability(samples, max_state, max_obser);
        toLog();
    }

    /**
     * 估计状态发射概率
     *
     * @param samples   训练样本集
     * @param max_state 状态的最大下标
     * @param max_obser 观测的最大下标
     */
    protected void estimateEmissionProbability(Collection<int[][]> samples, int max_state, int max_obser)
    {
        emission_probability = new float[max_state + 1][max_obser + 1];
        for (int[][] sample : samples)
        {
            for (int i = 0; i < sample[0].length; i++)
            {
                int o = sample[0][i];
                int s = sample[1][i];
                ++emission_probability[s][o];
            }
        }
        for (int i = 0; i < transition_probability.length; i++)
            normalize(emission_probability[i]);
    }

    /**
     * 利用极大似然估计转移概率
     *
     * @param samples   训练样本集
     * @param max_state 状态的最大下标，等于N-1
     */
    protected void estimateTransitionProbability(Collection<int[][]> samples, int max_state)
    {
        transition_probability = new float[max_state + 1][max_state + 1];
        for (int[][] sample : samples)
        {
            int prev_s = sample[1][0];
            for (int i = 1; i < sample[0].length; i++)
            {
                int s = sample[1][i];
                ++transition_probability[prev_s][s];
                prev_s = s;
            }
        }
        for (int i = 0; i < transition_probability.length; i++)
            normalize(transition_probability[i]);
    }

    /**
     * 估计初始状态概率向量
     *
     * @param samples   训练样本集
     * @param max_state 状态的最大下标
     */
    protected void estimateStartProbability(Collection<int[][]> samples, int max_state)
    {
        start_probability = new float[max_state + 1];
        for (int[][] sample : samples)
        {
            int s = sample[1][0];
            ++start_probability[s];
        }
        normalize(start_probability);
    }

    /**
     * 生成样本序列
     *
     * @param length 序列长度
     * @return 序列
     */
    public abstract int[][] generate(int length);


    /**
     * 生成样本序列
     *
     * @param minLength 序列最低长度
     * @param maxLength 序列最高长度
     * @param size      需要生成多少个
     * @return 样本序列集合
     */
    public List<int[][]> generate(int minLength, int maxLength, int size)
    {
        List<int[][]> samples = new ArrayList<int[][]>(size);
        for (int i = 0; i < size; i++)
        {
            samples.add(generate((int) (Math.floor(Math.random() * (maxLength - minLength)) + minLength)));
        }
        return samples;
    }

    /**
     * 预测（维特比算法）
     *
     * @param o 观测序列
     * @param s 预测状态序列（需预先分配内存）
     * @return 概率的对数，可利用 (float) Math.exp(maxScore) 还原
     */
    public abstract float predict(int[] o, int[] s);

    /**
     * 预测（维特比算法）
     *
     * @param o 观测序列
     * @param s 预测状态序列（需预先分配内存）
     * @return 概率的对数，可利用 (float) Math.exp(maxScore) 还原
     */
    public float predict(int[] o, Integer[] s)
    {
        int[] states = new int[s.length];
        float p = predict(o, states);
        for (int i = 0; i < states.length; i++)
        {
            s[i] = states[i];
        }
        return p;
    }

    public boolean similar(HiddenMarkovModel model)
    {
        if (!similar(start_probability, model.start_probability)) return false;
        for (int i = 0; i < transition_probability.length; i++)
        {
            if (!similar(transition_probability[i], model.transition_probability[i])) return false;
            if (!similar(emission_probability[i], model.emission_probability[i])) return false;
        }
        return true;
    }

    protected static boolean similar(float[] A, float[] B)
    {
        final float eta = 1e-2f;
        for (int i = 0; i < A.length; i++)
            if (Math.abs(A[i] - B[i]) > eta) return false;
        return true;
    }
}