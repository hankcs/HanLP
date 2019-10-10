/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-02-12 00:37</create-date>
 *
 * <copyright file="OpinionMining.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch12;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.dependency.IDependencyParser;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;

import java.io.IOException;
import java.util.List;

/**
 * 《自然语言处理入门》12.6 案例:基于依存句法树的意见抽取
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class OpinionMining
{
    public static void main(String[] args) throws IOException, ClassNotFoundException
    {
        IDependencyParser parser = new KBeamArcEagerDependencyParser();
        CoNLLSentence tree = parser.parse("电池非常棒，机身不长，长的是待机，但是屏幕分辨率不高。");
        System.out.println(tree);
        System.out.println("第一版");
        extactOpinion1(tree);
        System.out.println("第二版");
        extactOpinion2(tree);
        System.out.println("第三版");
        extactOpinion3(tree);
    }

    static void extactOpinion1(CoNLLSentence tree)
    {
        for (CoNLLWord word : tree)
            if (word.POSTAG.equals("NN") && word.DEPREL.equals("nsubj"))
                System.out.printf("%s = %s\n", word.LEMMA, word.HEAD.LEMMA);
    }

    static void extactOpinion2(CoNLLSentence tree)
    {
        for (CoNLLWord word : tree)
        {
            if (word.POSTAG.equals("NN") && word.DEPREL.equals("nsubj"))
            {
                if (tree.findChildren(word.HEAD, "neg").isEmpty())
                    System.out.printf("%s = %s\n", word.LEMMA, word.HEAD.LEMMA);
                else
                    System.out.printf("%s = 不%s\n", word.LEMMA, word.HEAD.LEMMA);
            }
        }
    }

    static void extactOpinion3(CoNLLSentence tree)
    {
        for (CoNLLWord word : tree)
        {
            if (word.POSTAG.equals("NN"))
            {
                if (word.DEPREL.equals("nsubj"))
                {
                    if (tree.findChildren(word.HEAD, "neg").isEmpty())
                        System.out.printf("%s = %s\n", word.LEMMA, word.HEAD.LEMMA);
                    else
                        System.out.printf("%s = 不%s\n", word.LEMMA, word.HEAD.LEMMA);
                }
                else if (word.DEPREL.equals("attr")) // ①属性
                {
                    List<CoNLLWord> top = tree.findChildren(word.HEAD, "top"); // ②主题
                    if (!top.isEmpty())
                        System.out.printf("%s = %s\n", word.LEMMA, top.get(0).LEMMA);
                }
            }
        }
    }
}
