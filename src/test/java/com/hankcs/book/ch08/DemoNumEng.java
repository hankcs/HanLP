/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-24 4:41 PM</create-date>
 *
 * <copyright file="DemoNumEng.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.dictionary.other.CharType;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;

/**
 * 《自然语言处理入门》8.2.3 基于规则的数词英文识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoNumEng
{
    public static void main(String[] args)
    {
        Segment segment = new ViterbiSegment();
        System.out.println(segment.seg("牛奶三〇〇克壹佰块"));
        System.out.println(segment.seg("牛奶300克100块"));
        System.out.println(segment.seg("牛奶300g100rmb"));
        // 演示自定义字符类型
        String text = "牛奶300~400g100rmb";
        System.out.println(segment.seg(text));
        CharType.type['~'] = CharType.CT_NUM;
        System.out.println(segment.seg(text));
    }
}
