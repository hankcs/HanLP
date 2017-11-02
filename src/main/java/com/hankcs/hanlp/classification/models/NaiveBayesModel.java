package com.hankcs.hanlp.classification.models;

import java.util.HashMap;
import java.util.Map;

/**
 * 储存学习过程中的数据
 */
public class NaiveBayesModel extends AbstractModel
{

    /**
     * 先验概率的对数值 log( P(c) )
     */
    public Map<Integer, Double> logPriors = new HashMap<Integer, Double>();

    /**
     * 似然对数值 log( P(x|c) )
     */
    public Map<Integer, Map<Integer, Double>> logLikelihoods = new HashMap<Integer, Map<Integer, Double>>();

    /**
     * 训练样本数
     */
    public int n = 0;
    /**
     * 类别数
     */
    public int c = 0;
    /**
     * 特征数
     */
    public int d = 0;
}