/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-30 8:36 PM</create-date>
 *
 * <copyright file="DemoExtractNewWord.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch09;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.mining.word.WordInfo;
import com.hankcs.hanlp.utility.TestUtility;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * 《自然语言处理入门》9.1 新词提取
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoExtractNewWord
{
    // 文本长度越大越好，试试四大名著？
    static final String HLM_PATH = TestUtility.ensureTestData("红楼梦.txt", "http://file.hankcs.com/corpus/红楼梦.zip");
    static final String XYJ_PATH = TestUtility.ensureTestData("西游记.txt", "http://file.hankcs.com/corpus/西游记.zip");
    static final String SHZ_PATH = TestUtility.ensureTestData("水浒传.txt", "http://file.hankcs.com/corpus/水浒传.zip");
    static final String SAN_PATH = TestUtility.ensureTestData("三国演义.txt", "http://file.hankcs.com/corpus/三国演义.zip");
    static final String WEIBO_PATH = TestUtility.ensureTestData("weibo-classification", "http://file.hankcs.com/corpus/weibo-classification.zip");

    public static void main(String[] args) throws IOException
    {
        extract(HLM_PATH);
        extract(XYJ_PATH);
        extract(SHZ_PATH);
        extract(SAN_PATH);
        testWeibo();

        // 更多参数
        List<WordInfo> wordInfoList = HanLP.extractWords(IOUtil.newBufferedReader(HLM_PATH), 100, true, 4, 0.0f, .5f, 100f);
        System.out.println(wordInfoList);
    }

    public static void testWeibo()
    {
        for (File folder : new File(WEIBO_PATH).listFiles())
        {
            System.out.println(folder.getName());
            StringBuilder sbText = new StringBuilder();
            for (File file : folder.listFiles())
            {
                sbText.append(IOUtil.readTxt(file.getPath()));
            }
            List<WordInfo> wordInfoList = HanLP.extractWords(sbText.toString(), 100);
            System.out.println(wordInfoList);
        }
    }

    private static void extract(String corpus) throws IOException
    {
        System.out.printf("%s 热词\n", corpus);
        List<WordInfo> wordInfoList = HanLP.extractWords(IOUtil.newBufferedReader(corpus), 100);
        System.out.println(wordInfoList);
//        System.out.printf("%s 新词\n", corpus);
//        wordInfoList = HanLP.extractWords(IOUtil.newBufferedReader(corpus), 100, true);
//        System.out.println(wordInfoList);
    }
}
