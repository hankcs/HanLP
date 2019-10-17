/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-21 7:30 PM</create-date>
 *
 * <copyright file="EasiestClassifier.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch05;

import com.hankcs.hanlp.model.perceptron.PerceptronNameGenderClassifier;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;

import java.util.LinkedList;
import java.util.List;

/**
 * 《自然语言处理入门》5.3 基于感知机的人名性别分类
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class CheapFeatureClassifier extends PerceptronNameGenderClassifier
{
    @Override
    protected List<Integer> extractFeature(String text, FeatureMap featureMap)
    {
        List<Integer> featureList = new LinkedList<Integer>();
        String givenName = extractGivenName(text);
        // 特征模板1：g[0]，与位置无关
        addFeature(givenName.substring(0, 1), featureMap, featureList);
        // 特征模板2：g[1]，与位置无关
        addFeature(givenName.substring(1), featureMap, featureList);
        return featureList;
    }
}
