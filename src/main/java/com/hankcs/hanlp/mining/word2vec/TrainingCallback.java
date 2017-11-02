/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-06-20 PM4:59</create-date>
 *
 * <copyright file="TrainingCallback.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.word2vec;

/**
 * @author hankcs
 */
public interface TrainingCallback
{
    /**
     * 语料加载中
     * @param percent 已加载的百分比（0-100）
     */
    void corpusLoading(float percent);

    /**
     * 语料加载完毕
     * @param vocWords 词表大小（不是词频，而是语料中有多少种词）
     * @param trainWords 实际训练用到的词的总词频（有些词被停用词过滤掉）
     * @param totalWords 全部词语的总词频
     */
    void corpusLoaded(int vocWords, int trainWords, int totalWords);

    /**
     * 训练过程的回调
     * @param alpha 学习率
     * @param progress 训练完成百分比（0-100）
     */
    void training(float alpha, float progress);
}
