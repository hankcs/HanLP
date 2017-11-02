/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016/1/29 17:59</create-date>
 *
 * <copyright file="ITextClassifier.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.classifiers;

import com.hankcs.hanlp.classification.corpus.Document;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.models.AbstractModel;

import java.io.IOException;
import java.util.Map;

/**
 * 文本分类器接口
 *
 * @author hankcs
 */
public interface IClassifier
{
    /**
     * 是否归一化分值为概率
     *
     * @param enable
     * @return
     */
    IClassifier enableProbability(boolean enable);

    /**
     * 预测分类
     *
     * @param text 文本
     * @return 所有分类对应的分值(或概率, 需要enableProbability)
     * @throws IllegalArgumentException 参数错误
     * @throws IllegalStateException    未训练模型
     */
    Map<String, Double> predict(String text) throws IllegalArgumentException, IllegalStateException;

    /**
     * 预测分类
     * @param document
     * @return
     */
    Map<String, Double> predict(Document document) throws IllegalArgumentException, IllegalStateException;

    /**
     * 预测分类
     * @param document
     * @return
     * @throws IllegalArgumentException
     * @throws IllegalStateException
     */
    double[] categorize(Document document) throws IllegalArgumentException, IllegalStateException;

    /**
     * 预测最可能的分类
     * @param document
     * @return
     * @throws IllegalArgumentException
     * @throws IllegalStateException
     */
    int label(Document document) throws IllegalArgumentException, IllegalStateException;

    /**
     * 预测最可能的分类
     * @param text 文本
     * @return 最可能的分类
     * @throws IllegalArgumentException
     * @throws IllegalStateException
     */
    String classify(String text) throws IllegalArgumentException, IllegalStateException;

    /**
     * 预测最可能的分类
     * @param document 一个结构化的文档(注意!这是一个底层数据结构,请谨慎操作)
     * @return 最可能的分类
     * @throws IllegalArgumentException
     * @throws IllegalStateException
     */
    String classify(Document document) throws IllegalArgumentException, IllegalStateException;

    /**
     * 训练模型
     *
     * @param trainingDataSet 训练数据集,用Map储存.键是分类名,值是一个数组,数组中每个元素都是一篇文档的内容.
     */
    void train(Map<String, String[]> trainingDataSet) throws IllegalArgumentException;

    /**
     * 训练模型
     *
     * @param folderPath  分类语料的根目录.目录必须满足如下结构:<br>
     *                    根目录<br>
     *                    ├── 分类A<br>
     *                    │   └── 1.txt<br>
     *                    │   └── 2.txt<br>
     *                    │   └── 3.txt<br>
     *                    ├── 分类B<br>
     *                    │   └── 1.txt<br>
     *                    │   └── ...<br>
     *                    └── ...<br>
     *                    文件不一定需要用数字命名,也不需要以txt作为后缀名,但一定需要是文本文件.
     * @param charsetName 文件编码
     * @throws IOException 任何可能的IO异常
     */
    void train(String folderPath, String charsetName) throws IOException;

    /**
     * 用UTF-8编码的语料训练模型
     *
     * @param folderPath  用UTF-8编码的分类语料的根目录.目录必须满足如下结构:<br>
     *                    根目录<br>
     *                    ├── 分类A<br>
     *                    │   └── 1.txt<br>
     *                    │   └── 2.txt<br>
     *                    │   └── 3.txt<br>
     *                    ├── 分类B<br>
     *                    │   └── 1.txt<br>
     *                    │   └── ...<br>
     *                    └── ...<br>
     *                    文件不一定需要用数字命名,也不需要以txt作为后缀名,但一定需要是文本文件.
     * @throws IOException 任何可能的IO异常
     */
    void train(String folderPath) throws IOException;

    /**
     * 训练模型
     * @param dataSet 训练数据集
     * @throws IllegalArgumentException 当数据集为空时,将抛出此异常
     */
    void train(IDataSet dataSet) throws IllegalArgumentException;

    /**
     * 获取训练后的模型,可用于序列化保存或预测.
     * @return 模型,null表示未训练
     */
    AbstractModel getModel();
}
