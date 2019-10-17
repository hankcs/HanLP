/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-02-06 21:15</create-date>
 *
 * <copyright file="DebugKBeamArcEagerDependencyParser.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch12;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.dependency.IDependencyParser;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.ArcEager;

import java.io.IOException;

/**
 * 《自然语言处理入门》12.4 基于转移的依存句法分析
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 * 请在{@link ArcEager#commitAction}中下一个断点，观察ArcEager转移系统
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DebugKBeamArcEagerDependencyParser
{
    public static void main(String[] args) throws IOException, ClassNotFoundException
    {
        IDependencyParser parser = new KBeamArcEagerDependencyParser();
        CoNLLSentence sentence = parser.parse("人吃鱼");
        System.out.println(sentence);
    }
}
