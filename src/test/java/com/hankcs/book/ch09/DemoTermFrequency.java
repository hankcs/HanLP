/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-09-16 11:59 PM</create-date>
 *
 * <copyright file="DemoTermFrequency.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch09;

import com.hankcs.hanlp.corpus.occurrence.TermFrequency;
import com.hankcs.hanlp.mining.word.TermFrequencyCounter;
import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;

import java.io.IOException;

/**
 * 《自然语言处理入门》9.2 关键词提取
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoTermFrequency
{
    public static void main(String[] args) throws IOException
    {
        TermFrequencyCounter counter = new TermFrequencyCounter();
//        counter.getSegment().enableIndexMode(true);
//        counter.setSegment(new PerceptronLexicalAnalyzer().enableIndexMode(true));
        counter.add("加油加油中国队!"); // 第一个文档
        counter.add("中国观众高呼加油中国"); // 第二个文档
        for (TermFrequency termFrequency : counter) // 遍历每个词与词频
            System.out.printf("%s=%d\n", termFrequency.getTerm(), termFrequency.getFrequency());
        System.out.println(counter.top(2)); // 取top N
    }
}
