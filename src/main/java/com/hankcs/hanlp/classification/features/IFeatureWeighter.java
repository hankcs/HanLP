package com.hankcs.hanlp.classification.features;

import java.io.Serializable;

/**
 * 词权重计算
 */
public interface IFeatureWeighter extends Serializable
{
    /**
     * 计算权重
     *
     * @param feature 词的id
     * @return 权重
     */
    double weight(int feature, int tf);
}