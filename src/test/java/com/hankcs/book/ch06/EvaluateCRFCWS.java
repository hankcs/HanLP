/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-26 3:14 PM</create-date>
 *
 * <copyright file="EvaluateCRFCWS.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch06;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer;
import com.hankcs.hanlp.model.crf.CRFSegmenter;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.CWSEvaluator;

import java.io.IOException;

import static com.hankcs.book.ch06.CrfppTrainHanLPLoad.CRF_MODEL_PATH;
import static com.hankcs.book.ch06.CrfppTrainHanLPLoad.CRF_MODEL_TXT_PATH;

/**
 * 《自然语言处理入门》6.4 HanLP 中的 CRF++ API
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class EvaluateCRFCWS
{
    public static Segment train(String corpus) throws IOException
    {
        CRFSegmenter segmenter = new CRFSegmenter(null);
        segmenter.train(corpus, CRF_MODEL_PATH);
        return new CRFLexicalAnalyzer(segmenter);
        // 训练完毕时，可传入txt格式的模型（不可传入CRF++的二进制模型，不兼容！）
//        return new CRFLexicalAnalyzer(CRF_MODEL_TXT_PATH).enableCustomDictionary(false);
    }

    public static void main(String[] args) throws IOException
    {
        Segment segment = train(MSR.TRAIN_PATH);
        System.out.println(CWSEvaluator.evaluate(segment, MSR.TEST_PATH, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)); // 标准化评测
    }
}
