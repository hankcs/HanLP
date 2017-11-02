/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/16 PM12:04</create-date>
 *
 * <copyright file="Demo.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.classification.classifiers.IClassifier;
import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier;
import com.hankcs.hanlp.classification.corpus.FileDataSet;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.corpus.MemoryDataSet;
import com.hankcs.hanlp.classification.statistics.evaluations.Evaluator;
import com.hankcs.hanlp.classification.statistics.evaluations.FMeasure;
import com.hankcs.hanlp.classification.tokenizers.BigramTokenizer;
import com.hankcs.hanlp.classification.tokenizers.HanLPTokenizer;

import java.io.IOException;

import static com.hankcs.demo.DemoTextClassification.CORPUS_FOLDER;


/**
 * 演示了分割训练集和测试集,进行更严谨的测试
 *
 * @author hankcs
 */
public class DemoTextClassificationFMeasure
{

    public static void main(String[] args) throws IOException
    {
        IDataSet trainingCorpus = new FileDataSet().                          // FileDataSet省内存，可加载大规模数据集
            setTokenizer(new HanLPTokenizer()).                               // 支持不同的ITokenizer，详见源码中的文档
            load(CORPUS_FOLDER, "UTF-8", 0.9);               // 前90%作为训练集
        IClassifier classifier = new NaiveBayesClassifier();
        classifier.train(trainingCorpus);
        IDataSet testingCorpus = new MemoryDataSet(classifier.getModel()).
            load(CORPUS_FOLDER, "UTF-8", -0.1);        // 后10%作为测试集
        // 计算准确率
        FMeasure result = Evaluator.evaluate(classifier, testingCorpus);
        System.out.println(result);
        // 搜狗文本分类语料库上的准确率与速度（两种不同的ITokenizer）
        // ITokenizer         F1      速度
        // HanLPTokenizer   97.04%  20833.33 doc/s
        // BigramTokenizer  96.82%  3521.13 doc/s
    }
}
