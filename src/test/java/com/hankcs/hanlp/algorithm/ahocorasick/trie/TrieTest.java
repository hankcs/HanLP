package com.hankcs.hanlp.algorithm.ahocorasick.trie;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.util.TreeMap;

public class TrieTest extends TestCase
{
    public void testHasKeyword() throws Exception
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        String[] keyArray = new String[]
            {
                "hers",
                "his",
                "she",
                "he"
            };
        for (String key : keyArray)
        {
            map.put(key, key);
        }
        Trie trie = new Trie();
        trie.addAllKeyword(map.keySet());
        for (String key : keyArray)
        {
            assertTrue(trie.hasKeyword(key));
        }
        assertTrue(trie.hasKeyword("ushers"));
        assertFalse(trie.hasKeyword("构建耗时"));
    }

    public void testParseText() throws Exception
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        String[] keyArray = new String[]
            {
                "hers",
                "his",
                "she",
                "he"
            };
        for (String key : keyArray)
        {
            map.put(key, key);
        }
        AhoCorasickDoubleArrayTrie<String> act = new AhoCorasickDoubleArrayTrie<String>();
        act.build(map);
//        act.debug();
        final String text = "uhers";
        act.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<String>()
        {
            @Override
            public void hit(int begin, int end, String value)
            {
//                System.out.printf("[%d:%d]=%s\n", begin, end, value);
                assertEquals(value, text.substring(begin, end));
            }
        });
    }


}