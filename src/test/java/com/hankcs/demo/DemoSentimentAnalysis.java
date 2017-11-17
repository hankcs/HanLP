/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/20 AM11:46</create-date>
 *
 * <copyright file="DemoAtFirstSight.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;


import com.hankcs.hanlp.classification.classifiers.IClassifier;
import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier;

import java.io.File;
import java.io.IOException;

/**
 * 第一个demo,演示文本分类最基本的调用方式
 *
 * @author hankcs
 */
public class DemoSentimentAnalysis
{
    /**
     * 中文情感挖掘语料-ChnSentiCorp 谭松波
     */
    public static final String CORPUS_FOLDER = "data/test/ChnSentiCorp情感分析酒店评论";

    public static void main(String[] args) throws IOException
    {
        IClassifier classifier = new NaiveBayesClassifier(); // 创建分类器，更高级的功能请参考IClassifier的接口定义
        classifier.train(CORPUS_FOLDER);                     // 训练后的模型支持持久化，下次就不必训练了
        predict(classifier, "前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！");
        predict(classifier, "结果大失所望，灯光昏暗，空间极其狭小，床垫质量恶劣，房间还伴着一股霉味。");
        predict(classifier, "可利用文本分类实现情感分析，效果还行");
    }

    private static void predict(IClassifier classifier, String text)
    {
        System.out.printf("《%s》 情感极性是 【%s】\n", text, classifier.classify(text));
    }

    static
    {
        File corpusFolder = new File(CORPUS_FOLDER);
        if (!corpusFolder.exists() || !corpusFolder.isDirectory())
        {
            System.err.println("没有文本分类语料，请阅读IClassifier.train(java.lang.String)中定义的语料格式、准备语料");
            System.exit(1);
        }
    }
}
