/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/24 20:01</create-date>
 *
 * <copyright file="TestStopWordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.MDAG.MDAGSet;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.dictionary.stopword.StopWordDictionary;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.File;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class TestStopWordDictionary extends TestCase
{
    public void testContains() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(CoreStopWordDictionary.contains("这就是说"));
    }

    public void testContainsSomeWords() throws Exception
    {
        assertEquals(true, CoreStopWordDictionary.contains("可以"));
    }

    public void testMDAG() throws Exception
    {
        List<String> wordList = new LinkedList<String>();
        wordList.add("zoo");
        wordList.add("hello");
        wordList.add("world");
        MDAGSet set = new MDAGSet(wordList);
        set.add("bee");
        assertEquals(true, set.contains("bee"));
        set.remove("bee");
        assertEquals(false, set.contains("bee"));
    }

    public void testRemoveDuplicateEntries() throws Exception
    {
        StopWordDictionary dictionary = new StopWordDictionary(new File(HanLP.Config.CoreStopWordDictionaryPath));
        BufferedWriter bw = IOUtil.newBufferedWriter(HanLP.Config.CoreStopWordDictionaryPath);
        for (String word : dictionary)
        {
            bw.write(word);
            bw.newLine();
        }
        bw.close();
    }
}
