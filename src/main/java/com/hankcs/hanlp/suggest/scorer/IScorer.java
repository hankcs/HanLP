/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/5 15:41</create-date>
 *
 * <copyright file="IScorer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest.scorer;

import java.util.Map;

/**
 * 打分接口
 * @author hankcs
 */
public interface IScorer
{
    /**
     * 计算分值
     * @param outerSentence 外部句子
     * @return key为分值x，value为分值等于x的一系列句子
     */
    Map<String, Double> computeScore(String outerSentence);

    /**
     * 输入一个候选句子
     * @param sentence
     */
    void addSentence(String sentence);

    /**
     * 清空该推荐器中的所有句子
     */
    void removeAllSentences();
}
