/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-07 11:38 PM</create-date>
 *
 * <copyright file="DemoAdjustModel.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch03;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.Segment;

import static com.hankcs.book.ch03.DemoNgramSegment.MSR_MODEL_PATH;
import static com.hankcs.book.ch03.DemoNgramSegment.loadBigram;

/**
 * 《自然语言处理入门》3.5 评测
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoAdjustModel
{
    public static void main(String[] args)
    {
        Segment segment = loadBigram(MSR_MODEL_PATH, false, false);
        assert CoreDictionary.contains("管道");
        String text = "北京输气管道工程";
        HanLP.Config.enableDebug();
        System.out.println(segment.seg(text));
    }
}
