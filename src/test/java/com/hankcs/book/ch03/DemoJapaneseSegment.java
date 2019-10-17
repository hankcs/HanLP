/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-07 10:50 PM</create-date>
 *
 * <copyright file="DemoJapaneseSegment.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch03;

import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.utility.TestUtility;

import static com.hankcs.book.ch03.DemoNgramSegment.loadBigram;
import static com.hankcs.book.ch03.DemoNgramSegment.trainBigram;

/**
 * 《自然语言处理入门》3.6 日语分词
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoJapaneseSegment
{
    static final String CORPUS_PATH = TestUtility.ensureTestData("jpcorpus", "http://file.hankcs.com/corpus/jpcorpus.zip") + "/ja_gsd-ud-train.txt";
    static final String MODEL_PATH = "data/test/jpcorpus/jp_bigram";

    public static void main(String[] args)
    {
        trainBigram(CORPUS_PATH, MODEL_PATH);
        Segment segment = loadBigram(MODEL_PATH, false, true); // data/test/jpcorpus/jp_bigram
        System.out.println(segment.seg("自然言語処理入門という本が面白いぞ！"));
    }
}
