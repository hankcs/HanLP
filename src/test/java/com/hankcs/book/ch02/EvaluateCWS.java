/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-03 下午5:17</create-date>
 *
 * <copyright file="EvaluateCWS.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

/**
 * 《自然语言处理入门》2.9 准确率评测
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class EvaluateCWS
{
    public static void main(String[] args) throws IOException
    {
        String trainWords = MSR.TRAIN_WORDS;
        Segment segment = new DoubleArrayTrieSegment(trainWords)
            .enablePartOfSpeechTagging(true);
        CWSEvaluator.Result result = CWSEvaluator.evaluate(segment, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, trainWords);
        System.out.println(result);
    }
}
