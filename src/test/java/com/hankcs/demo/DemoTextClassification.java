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
public class DemoTextClassification
{
    /**
     * 搜狗文本分类语料库5个类目，每个类目下1000篇文章，共计5000篇文章
     */
    public static final String CORPUS_FOLDER = "data/test/搜狗文本分类语料库迷你版";

    public static void main(String[] args) throws IOException
    {
        IClassifier classifier = new NaiveBayesClassifier(); // 创建分类器，更高级的功能请参考IClassifier的接口定义
        classifier.train(CORPUS_FOLDER);                     // 训练后的模型支持持久化，下次就不必训练了
        predict(classifier, "NBA-哈登31分火箭大胜");
        predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后");
        predict(classifier, "研究生考录模式亟待进一步专业化");
        predict(classifier, "如果真想用食物解压,建议可以食用燕麦");
        predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题");
    }

    private static void predict(IClassifier classifier, String text)
    {
        System.out.printf("《%s》 属于分类 【%s】\n", text, classifier.classify(text));
    }

    static
    {
        File corpusFolder = new File(CORPUS_FOLDER);
        if (!corpusFolder.exists() || !corpusFolder.isDirectory())
        {
            System.err.println("没有文本分类语料，请阅读IClassifier.train(java.lang.String)中定义的语料格式与语料下载：" +
            "https://github.com/hankcs/HanLP/wiki/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E4%B8%8E%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90");
            System.exit(1);
        }
    }
}
