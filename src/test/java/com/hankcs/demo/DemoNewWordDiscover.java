/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-11-02 11:21</create-date>
 *
 * <copyright file="Demo.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.mining.word.WordInfo;

import java.io.IOException;
import java.util.List;

/**
 * 词语提取、新词发现
 *
 * @author hankcs
 */
public class DemoNewWordDiscover
{
    public static void main(String[] args) throws IOException
    {
        // 文本长度越大越好，试试红楼梦？
        List<WordInfo> wordInfoList = HanLP.extractWords(IOUtil.newBufferedReader("data/test/红楼梦.txt"), 100);
        System.out.println(wordInfoList);
    }
}
