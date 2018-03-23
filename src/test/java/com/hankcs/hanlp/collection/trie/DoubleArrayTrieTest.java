package com.hankcs.hanlp.collection.trie;

import junit.framework.TestCase;

import java.util.TreeMap;

public class DoubleArrayTrieTest extends TestCase
{
    public void testLongestSearcher() throws Exception
    {
        TreeMap<String, String> buildFrom = new TreeMap<String, String>();
        String[] keys = new String[]{"he", "her", "his"};
        for (String key : keys)
        {
            buildFrom.put(key, key);
        }
        DoubleArrayTrie<String> trie = new DoubleArrayTrie<String>(buildFrom);
        String text = "her3he6his! ";
        DoubleArrayTrie<String>.LongestSearcher searcher = trie.getLongestSearcher(text.toCharArray(), 0);
        while (searcher.next())
        {
            System.out.printf("[%d, %d)=%s\n", searcher.begin, searcher.begin + searcher.length, searcher.value);
        }
    }
}