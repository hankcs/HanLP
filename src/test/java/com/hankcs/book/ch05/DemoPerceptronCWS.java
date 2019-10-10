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

import com.hankcs.hanlp.HanLP;
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
public class DemoPerceptronCWS
{
    public static void main(String[] args) throws IOException
    {
        Segment segment = train();
        String[] sents = new String[]{
            "王思斌，男，１９４９年１０月生。",
            "山东桓台县起凤镇穆寨村妇女穆玲英",
            "现为中国艺术研究院中国文化研究所研究员。",
            "我们的父母重男轻女",
            "北京输气管道工程",
        };
        for (String sent : sents)
        {
            System.out.println(segment.seg(sent));
        }
//        trainUncompressedModel();
    }

    public static Segment train() throws IOException
    {
        LinearModel model = new CWSTrainer().train(MSR.TRAIN_PATH, MSR.MODEL_PATH).getModel(); // 训练模型
        Segment segment = new PerceptronLexicalAnalyzer(model).enableCustomDictionary(false); // 创建分词器
        System.out.println(CWSEvaluator.evaluate(segment, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)); // 标准化评测
        return segment;
    }

    private static Segment trainUncompressedModel() throws IOException
    {
        LinearModel model = new CWSTrainer().train(MSR.TRAIN_PATH, MSR.TRAIN_PATH, MSR.MODEL_PATH, 0., 10, 8).getModel();
        model.save(MSR.MODEL_PATH, model.featureMap.entrySet(), 0, true); // 最后一个参数指定导出txt
        return new PerceptronLexicalAnalyzer(model).enableCustomDictionary(false);
    }

    static
    {
        HanLP.Config.ShowTermNature = false;
    }
}
