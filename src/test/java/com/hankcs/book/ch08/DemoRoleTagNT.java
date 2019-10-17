/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-27 11:46 AM</create-date>
 *
 * <copyright file="DemoRoleTagNT.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.NSDictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.NTDictionaryMaker;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Segment;

import java.io.IOException;

/**
 * 《自然语言处理入门》8.4.3 基于角色标注的机构名识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoRoleTagNT
{
    public static final String MODEL = "data/test/nt";

    public static void main(String[] args)
    {
        train(PKU.PKU199801, MODEL);
        test(MODEL);
    }

    private static void train(String corpus, String model)
    {
        EasyDictionary dictionary = EasyDictionary.create(HanLP.Config.CoreDictionaryPath); // 核心词典
        NTDictionaryMaker maker = new NTDictionaryMaker(dictionary); // 训练模块
        maker.train(corpus); // 在语料库上训练
        maker.saveTxtTo(model); // 输出HMM到txt
    }

    private static Segment load(String model)
    {
        HanLP.Config.PlaceDictionaryPath = model + ".txt"; // data/test/ns.txt
        HanLP.Config.PlaceDictionaryTrPath = model + ".tr.txt"; // data/test/ns.tr.txt
        Segment segment = new DijkstraSegment(); // 该分词器便于调试
        return segment.enableOrganizationRecognize(true).enableCustomDictionary(false);
    }

    private static void test(String model)
    {
        Segment segment = load(model);
        HanLP.Config.enableDebug();
        System.out.println(segment.seg("温州黄鹤皮革制造有限公司是由黄先生创办的企业"));
    }
}
