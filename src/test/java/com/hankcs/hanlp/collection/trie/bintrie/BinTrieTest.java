package com.hankcs.hanlp.collection.trie.bintrie;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import junit.framework.TestCase;

public class BinTrieTest extends TestCase
{
    public void testParseText() throws Exception
    {
        BinTrie<String> trie = new BinTrie<String>();
        String[] keys = new String[]{"he", "her", "his"};
        for (String key : keys)
        {
            trie.put(key, key);
        }
        final String text = " her4he7his ";
        AhoCorasickDoubleArrayTrie.IHit<String> processor = new AhoCorasickDoubleArrayTrie.IHit<String>()
        {
            @Override
            public void hit(int begin, int end, String value)
            {
                System.out.printf("[%d, %d)=%s\n", begin, end, value);
                assertEquals(value, text.substring(begin, end));
            }
        };
//        trie.parseLongestText(text, processor);
        trie.parseText(text, processor);
    }
}