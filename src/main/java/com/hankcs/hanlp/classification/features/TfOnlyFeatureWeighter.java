package com.hankcs.hanlp.classification.features;

/**
 * 仅仅使用TF的权重计算方式
 */
public class TfOnlyFeatureWeighter implements IFeatureWeighter
{
    public double weight(int feature, int tf)
    {
        return tf;
    }
}