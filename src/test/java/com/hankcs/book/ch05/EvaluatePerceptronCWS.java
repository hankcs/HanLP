/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-22 3:15 PM</create-date>
 *
 * <copyright file="DemoPerceptronCWS.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch05;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.model.perceptron.CWSTrainer;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

/**
 * 《自然语言处理入门》5.6 基于结构化感知机的中文分词
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class EvaluatePerceptronCWS
{
    public static Segment trainStructuredPerceptron() throws IOException
    {
        LinearModel model = new CWSTrainer().train(MSR.TRAIN_PATH, MSR.TRAIN_PATH, MSR.MODEL_PATH, 0., 10, 8).getModel();
        return new PerceptronLexicalAnalyzer(model).enableCustomDictionary(false);
    }

    public static Segment trainAveragedPerceptron() throws IOException
    {
        // 线程数为1时自动用平均感知机算法
        LinearModel model = new CWSTrainer().train(MSR.TRAIN_PATH, MSR.TRAIN_PATH, MSR.MODEL_PATH, 0., 10, 1).getModel();
        return new PerceptronLexicalAnalyzer(model).enableCustomDictionary(false);
    }

    public static void main(String[] args) throws IOException
    {
        System.out.println("结构化感知机");
        System.out.println(CWSEvaluator.evaluate(trainStructuredPerceptron(), MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS));
        System.out.println("平均感知机");
        System.out.println(CWSEvaluator.evaluate(trainAveragedPerceptron(), MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS));
    }
}
