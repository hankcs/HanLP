package com.hankcs.hanlp.collection.AhoCorasick;

import com.hankcs.hanlp.algorithm.ahocorasick.trie.Emit;
import com.hankcs.hanlp.algorithm.ahocorasick.trie.Trie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.util.*;

public class AhoCorasickDoubleArrayTrieTest extends TestCase
{

    public void testTwoAC() throws Exception
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        IOUtil.LineIterator iterator = new IOUtil.LineIterator("data/dictionary/CoreNatureDictionary.mini.txt");
        while (iterator.hasNext())
        {
            String line = iterator.next().split("\\s")[0];
            map.put(line, line);
        }

        Trie trie = new Trie();
        trie.addAllKeyword(map.keySet());
        AhoCorasickDoubleArrayTrie<String> act = new AhoCorasickDoubleArrayTrie<String>();
        act.build(map);

        for (String key : map.keySet())
        {
            Collection<Emit> emits = trie.parseText(key);
            Set<String> otherSet = new HashSet<String>();
            for (Emit emit : emits)
            {
                otherSet.add(emit.getKeyword() + emit.getEnd());
            }

            List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> entries = act.parseText(key);
            Set<String> mySet = new HashSet<String>();
            for (AhoCorasickDoubleArrayTrie<String>.Hit<String> entry : entries)
            {
                mySet.add(entry.value + (entry.end - 1));
            }

            assertEquals(otherSet, mySet);
        }
    }

    /**
     * 测试构建和匹配，使用《我的团长我的团》.txt作为测试数据，并且判断匹配是否正确
     * @throws Exception
     */
//    public void testSegment() throws Exception
//    {
//        TreeMap<String, String> map = new TreeMap<String, String>();
//        IOUtil.LineIterator iterator = new IOUtil.LineIterator("data/dictionary/CoreNatureDictionary.txt");
//        while (iterator.hasNext())
//        {
//            String line = iterator.next().split("\\s")[0];
//            map.put(line, line);
//        }
//
//        Trie trie = new Trie();
//        trie.addAllKeyword(map.keySet());
//        AhoCorasickDoubleArrayTrie<String> act = new AhoCorasickDoubleArrayTrie<String>();
//        long timeMillis = System.currentTimeMillis();
//        act.build(map);
//        System.out.println("构建耗时：" + (System.currentTimeMillis() - timeMillis) + " ms");
//
//        LinkedList<String> lineList = IOUtil.readLineList("D:\\Doc\\语料库\\《我的团长我的团》.txt");
//        timeMillis = System.currentTimeMillis();
//        for (String sentence : lineList)
//        {
////            System.out.println(sentence);
//            List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> entryList = act.parseText(sentence);
//            for (AhoCorasickDoubleArrayTrie<String>.Hit<String> entry : entryList)
//            {
//                int end = entry.end;
//                int start = entry.begin;
////                System.out.printf("[%d:%d]=%s\n", start, end, entry.value);
//
//                assertEquals(sentence.substring(start, end), entry.value);
//            }
//        }
//        System.out.printf("%d ms\n", System.currentTimeMillis() - timeMillis);
//    }
}