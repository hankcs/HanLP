/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-29 上午9:35</create-date>
 *
 * <copyright file="DemoDoubleArrayTrieSegment.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.IOException;

/**
 * 《自然语言处理入门》2.8 HanLP的词典分词实现
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoDoubleArrayTrieSegment
{
    public static void main(String[] args) throws IOException
    {
        HanLP.Config.ShowTermNature = false; // 分词结果不显示词性
        // 默认加载配置文件指定的 CoreDictionaryPath
        DoubleArrayTrieSegment segment = new DoubleArrayTrieSegment();
        System.out.println(segment.seg("江西鄱阳湖干枯，中国最大淡水湖变成大草原"));
        // 也支持加载自己的词典
        String dict1 = "data/dictionary/CoreNatureDictionary.mini.txt";
        String dict2 = "data/dictionary/custom/上海地名.txt ns";
        segment = new DoubleArrayTrieSegment(dict1, dict2);
        System.out.println(segment.seg("上海市虹口区大连西路550号SISU"));

        segment.enablePartOfSpeechTagging(true);    // 激活数词和英文识别
        HanLP.Config.ShowTermNature = true;         // 顺便观察一下词性
        System.out.println(segment.seg("上海市虹口区大连西路550号SISU"));

        for (Term term : segment.seg("上海市虹口区大连西路550号SISU"))
        {
            System.out.printf("单词:%s 词性:%s\n", term.word, term.nature);
        }
    }
}
