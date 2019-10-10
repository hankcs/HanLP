/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-07 3:41 PM</create-date>
 *
 * <copyright file="EvaluateBigram.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch03;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

import static com.hankcs.book.ch03.DemoNgramSegment.*;
import static com.hankcs.hanlp.seg.common.CWSEvaluator.evaluate;

/**
 * 《自然语言处理入门》3.5 评测
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class EvaluateBigram
{

    public static void main(String[] args) throws IOException
    {
        trainBigram(MSR.TRAIN_PATH, MSR_MODEL_PATH);
        Segment segment = loadBigram(MSR_MODEL_PATH);
        CWSEvaluator.Result result = evaluate(segment, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS);
        System.out.println(result);
    }
}