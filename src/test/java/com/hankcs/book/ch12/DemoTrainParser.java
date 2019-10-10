/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2019-02-08 01:57</create-date>
 *
 * <copyright file="DemoTrainParser.java">
 * Copyright (c) 2019, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.book.ch12;

import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.utility.TestUtility;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * 《自然语言处理入门》12.5 依存句法分析 API
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoTrainParser
{
    public static String CTB_ROOT = TestUtility.ensureTestData("ctb8.0-dep", "http://file.hankcs.com/corpus/ctb8.0-dep.zip");
    public static String CTB_TRAIN = CTB_ROOT + "/train.conll";
    public static String CTB_DEV = CTB_ROOT + "/dev.conll";
    public static String CTB_TEST = CTB_ROOT + "/test.conll";
    public static String CTB_MODEL = CTB_ROOT + "/ctb.bin";
    public static String BROWN_CLUSTER = TestUtility.ensureTestData("wiki-cn-cluster.txt", "http://file.hankcs.com/corpus/wiki-cn-cluster.zip");

    public static void main(String[] args) throws IOException, ClassNotFoundException, ExecutionException, InterruptedException
    {
        KBeamArcEagerDependencyParser parser = KBeamArcEagerDependencyParser.train(CTB_TRAIN, CTB_DEV, BROWN_CLUSTER, CTB_MODEL);
        System.out.println(parser.parse("人吃鱼"));
        double[] score = parser.evaluate(CTB_TEST);
        System.out.printf("UAS=%.1f LAS=%.1f\n", score[0], score[1]);
    }
}
