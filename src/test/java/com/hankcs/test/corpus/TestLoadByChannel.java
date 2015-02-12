/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/9 20:51</create-date>
 *
 * <copyright file="TestLoadByChannel.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestLoadByChannel extends TestCase
{
    public void testLoad() throws Exception
    {
        DoubleArrayTrie<Integer> trie = new DoubleArrayTrie<Integer>();
        trie.load("data/dictionary/CoreNatureDictionary.txt.trie.dat", new Integer[0]);
    }

    public void testHasNext() throws Exception
    {
        IOUtil.LineIterator iterator = IOUtil.readLine("data/test/other/f.txt");
        while (iterator.hasNext())
        {
            System.out.println(iterator.next());
        }
    }

    public void testNext() throws Exception
    {
        String line;
        IOUtil.LineIterator iterator = IOUtil.readLine("data/test/other/f.txt");
        while ((line = iterator.next()) != null)
        {
            System.out.println(line);
        }
    }

    public void testUTFString() throws Exception
    {

    }
}
