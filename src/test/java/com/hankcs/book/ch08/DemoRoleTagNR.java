/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-24 8:07 PM</create-date>
 *
 * <copyright file="DemoRoleTag.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.NRDictionaryMaker;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Segment;

import java.io.IOException;

/**
 * 《自然语言处理入门》8.4.1 基于角色标注的中国人名识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoRoleTagNR
{
    public static final String MODEL = "data/test/nr";

    public static void main(String[] args)
    {
        demoNR();
        trainOneSentence();
        train(PKU.PKU199801, MODEL);
        test();
    }

    private static void trainOneSentence()
    {
        EasyDictionary dictionary = EasyDictionary.create(HanLP.Config.CoreDictionaryPath); // 核心词典
        NRDictionaryMaker maker = new NRDictionaryMaker(dictionary); // 训练模块
        maker.verbose = true; // 调试输出
        maker.learn(Sentence.create("这里/r 有/v 关天培/nr 的/u 有关/vn 事迹/n 。/w")); // 学习一个句子
        maker.saveTxtTo(MODEL); // 输出HMM到txt
    }

    private static void train(String corpus, String model)
    {
        EasyDictionary dictionary = EasyDictionary.create(HanLP.Config.CoreDictionaryPath); // 核心词典
        NRDictionaryMaker maker = new NRDictionaryMaker(dictionary); // 训练模块
        maker.train(corpus); // 在语料库上训练
        maker.saveTxtTo(model); // 输出HMM到txt
    }

    private static Segment load(String model)
    {
        HanLP.Config.PersonDictionaryPath = model + ".txt"; // data/test/nr.txt
        HanLP.Config.PersonDictionaryTrPath = model + ".tr.txt"; // data/test/nr.tr.txt
        Segment segment = new DijkstraSegment(); // 该分词器便于调试
        return segment;
    }

    private static void test()
    {
        Segment segment = load(MODEL);
        HanLP.Config.enableDebug();
        System.out.println(segment.seg("龚学平等领导"));
    }

    private static void demoNR()
    {
        HanLP.Config.enableDebug();
        Segment segment = new DijkstraSegment();
        System.out.println(segment.seg("王国维和服务员"));
    }
}
