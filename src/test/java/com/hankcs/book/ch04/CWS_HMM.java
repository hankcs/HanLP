/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-13 3:11 PM</create-date>
 *
 * <copyright file="CWS_HMM.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch04;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel;
import com.hankcs.hanlp.model.hmm.HMMSegmenter;
import com.hankcs.hanlp.model.hmm.HiddenMarkovModel;
import com.hankcs.hanlp.model.hmm.SecondOrderHiddenMarkovModel;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

/**
 * 《自然语言处理入门》4.6 隐马尔可夫模型应用于中文分词
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 * 演示一阶和二阶隐马尔可夫模型用于序列标注问题之中文分词
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class CWS_HMM
{
    public static void main(String[] args) throws IOException
    {
        trainAndEvaluate(new FirstOrderHiddenMarkovModel());
        trainAndEvaluate(new SecondOrderHiddenMarkovModel());
    }

    public static void trainAndEvaluate(HiddenMarkovModel model) throws IOException
    {
        Segment hmm = trainHMM(model);
        CWSEvaluator.Result result = CWSEvaluator.evaluate(hmm, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS);
        System.out.println(result);
    }

    private static Segment trainHMM(HiddenMarkovModel model) throws IOException
    {
        HMMSegmenter segmenter = new HMMSegmenter(model);
        segmenter.train(MSR.TRAIN_PATH);
        System.out.println(segmenter.segment("商品和服务"));
        return segmenter.toSegment();
    }
}
