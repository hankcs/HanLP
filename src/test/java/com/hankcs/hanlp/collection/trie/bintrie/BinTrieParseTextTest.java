package com.hankcs.hanlp.collection.trie.bintrie;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

public class BinTrieParseTextTest {


    private final String[] words = new String[]{"溜", "儿", "溜儿", "一溜儿", "一溜"};
    private BinTrie<Integer> trie;

    @Before
    public void setup() {
        this.trie = new BinTrie<Integer>();
        /*构建一个简单的词典， 从 core dict 文件中扣出的一部分*/
        for (int i = 0; i < words.length; i++) {
            this.trie.put(words[i], i);
        }
    }


    @Test
    public void testFullParse() {
        assertFullParse("一溜儿");
        assertFullParse("一溜儿    ");
        assertFullParse("一溜儿 ");
    }

    private void assertFullParse(String text) {
        Set<String> result = parseText(text);
        /*确保每个词都被分出来了*/
        for (String word : words) {
            Assert.assertTrue(result.contains(word));
        }
    }


    private Set<String> parseText(final String text) {
        final Set<String> result = new HashSet<String>(words.length);
        trie.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<Integer>() {
            @Override
            public void hit(int begin, int end, Integer value) {
                result.add(text.substring(begin, end));
            }
        });

        return result;
    }


}
