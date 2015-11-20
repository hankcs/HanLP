/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/7/14 11:41</create-date>
 *
 * <copyright file="TestBinTrie.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.test.algorithm;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestBinTrie extends TestCase
{
    public void testPut() throws Exception
    {
        BinTrie<Boolean> trie = new BinTrie<Boolean>();
        trie.put("加入", true);
        trie.put("加入", false);

        assertEquals(new Boolean(false), trie.get("加入"));
    }

    public void testArrayIndexOutOfBoundsException() throws Exception
    {
        BinTrie<Boolean> trie = new BinTrie<Boolean>();
        trie.put(new char[]{'\uffff'}, true);
    }
}
