/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-01-06 10:15 PM</create-date>
 *
 * <copyright file="DemoTextClassificationFMeasure.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch11;

import com.hankcs.hanlp.classification.classifiers.IClassifier;
import com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier;
import com.hankcs.hanlp.classification.corpus.FileDataSet;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.corpus.MemoryDataSet;
import com.hankcs.hanlp.classification.statistics.evaluations.Evaluator;
import com.hankcs.hanlp.classification.statistics.evaluations.FMeasure;
import com.hankcs.hanlp.classification.tokenizers.BigramTokenizer;
import com.hankcs.hanlp.classification.tokenizers.HanLPTokenizer;
import com.hankcs.hanlp.classification.tokenizers.ITokenizer;

import java.io.IOException;

import static com.hankcs.demo.DemoTextClassification.CORPUS_FOLDER;

/**
 * 《自然语言处理入门》11.6 标准化评测
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoTextClassificationFMeasure
{
    public static void main(String[] args) throws IOException
    {
        evaluate(new NaiveBayesClassifier(), new HanLPTokenizer());
        evaluate(new NaiveBayesClassifier(), new BigramTokenizer());
        // 需要引入 https://github.com/hankcs/text-classification-svm ，或者将下列代码复制到该项目运行
        // evaluate(new NaiveBayesClassifier(), new HanLPTokenizer());
        // evaluate(new NaiveBayesClassifier(), new BigramTokenizer());
    }

    public static void evaluate(IClassifier classifier, ITokenizer tokenizer) throws IOException
    {
        IDataSet trainingCorpus = new FileDataSet().                          // FileDataSet省内存，可加载大规模数据集
            setTokenizer(tokenizer).                               // 支持不同的ITokenizer，详见源码中的文档
            load(CORPUS_FOLDER, "UTF-8", 0.9);               // 前90%作为训练集
        classifier.train(trainingCorpus);
        IDataSet testingCorpus = new MemoryDataSet(classifier.getModel()).
            load(CORPUS_FOLDER, "UTF-8", -0.1);        // 后10%作为测试集
        // 计算准确率
        FMeasure result = Evaluator.evaluate(classifier, testingCorpus);
        System.out.println(classifier.getClass().getSimpleName() + "+" + tokenizer.getClass().getSimpleName());
        System.out.println(result);
    }
}
