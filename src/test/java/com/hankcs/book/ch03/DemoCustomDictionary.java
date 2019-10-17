/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-07 2:24 PM</create-date>
 *
 * <copyright file="DemoCustomDictionary.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch03;

import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;

/**
 * 《自然语言处理入门》3.4 预测
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoCustomDictionary
{
    public static void main(String[] args)
    {
        Segment segment = new ViterbiSegment();
        final String sentence = "社会摇摆简称社会摇";
        segment.enableCustomDictionary(false);
        System.out.println("不挂载词典：" + segment.seg(sentence));
        CustomDictionary.insert("社会摇", "nz 100");
        segment.enableCustomDictionary(true);
        System.out.println("低优先级词典：" + segment.seg(sentence));
        segment.enableCustomDictionaryForcing(true);
        System.out.println("高优先级词典：" + segment.seg(sentence));
    }
}
