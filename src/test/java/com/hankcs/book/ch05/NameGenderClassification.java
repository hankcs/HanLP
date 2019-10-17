/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-21 5:38 PM</create-date>
 *
 * <copyright file="NameGenderClassification.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch05;

import com.hankcs.hanlp.model.perceptron.PerceptronNameGenderClassifier;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;

import java.io.IOException;

import static com.hankcs.hanlp.model.perceptron.PerceptronNameGenderClassifierTest.*;

/**
 * 《自然语言处理入门》5.3 基于感知机的人名性别分类
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class NameGenderClassification
{
    public static void main(String[] args) throws IOException
    {
        trainAndEvaluate("简单特征模板", new CheapFeatureClassifier(), false);
        trainAndEvaluate("简单特征模板", new CheapFeatureClassifier(), true);

        trainAndEvaluate("标准特征模板", new PerceptronNameGenderClassifier(), false);
        trainAndEvaluate("标准特征模板", new PerceptronNameGenderClassifier(), true);

        trainAndEvaluate("复杂特征模板", new RichFeatureClassifier(), false);
        trainAndEvaluate("复杂特征模板", new RichFeatureClassifier(), true);
    }

    private static void trainAndEvaluate(String template, PerceptronNameGenderClassifier classifier, boolean averagePerceptron) throws IOException
    {
        String algorithm = averagePerceptron ? "平均感知机算法" : "朴素感知机算法";
        System.out.println("训练集准确率：" + classifier.train(TRAINING_SET, 10, averagePerceptron));
        LinearModel model = classifier.getModel();
        System.out.printf("特征数量：%d\n", model.parameter.length);
        System.out.printf("%s+%s 测试集准确率：%s\n", algorithm, template, classifier.evaluate(TESTING_SET));
    }
}
