package com.hankcs.hanlp.collection.trie;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;
import org.junit.Assert;

import java.util.TreeMap;

public class DoubleArrayTrieTest extends TestCase
{
    public void testDatFromFile() throws Exception
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        IOUtil.LineIterator iterator = new IOUtil.LineIterator("data/dictionary/CoreNatureDictionary.mini.txt");
        while (iterator.hasNext())
        {
            String line = iterator.next();
            map.put(line, line);
        }
        DoubleArrayTrie<String> trie = new DoubleArrayTrie<String>();
        trie.build(map);
        for (String key : map.keySet())
        {
            assertEquals(key, trie.get(key));
        }

        trie.build(map);
        for (String key : map.keySet())
        {
            assertEquals(key, trie.get(key));
        }
    }

    public void testGet() throws Exception
    {
    }

    public void testLongestSearcher() throws Exception
    {
        TreeMap<String, String> buildFrom = new TreeMap<String, String>();
        String[] keys = new String[]{"he", "her", "his"};
        for (String key : keys)
        {
            buildFrom.put(key, key);
        }
        DoubleArrayTrie<String> trie = new DoubleArrayTrie<String>(buildFrom);
        String text = "her3he6his-hers! ";
        DoubleArrayTrie<String>.LongestSearcher searcher = trie.getLongestSearcher(text.toCharArray(), 0);
        while (searcher.next())
        {
//            System.out.printf("[%d, %d)=%s\n", searcher.begin, searcher.begin + searcher.length, searcher.value);
            assertEquals(searcher.value, text.substring(searcher.begin, searcher.begin + searcher.length));
        }
    }

    public void testLongestSearcherWithNullValue() {
        TreeMap<String, String> buildFrom = new TreeMap<String, String>();
        TreeMap<String, String> buildFromValueNull = new TreeMap<String, String>();
        String[] keys = new String[]{"he", "her", "his"};
        for (String key : keys) {
            buildFrom.put(key, key);
            buildFromValueNull.put(key, null);
        }
        DoubleArrayTrie<String> trie = new DoubleArrayTrie<String>(buildFrom);
        DoubleArrayTrie<String> trieValueNull = new DoubleArrayTrie<String>(buildFromValueNull);

        String text = "her3he6his-hers! ";

        DoubleArrayTrie<String>.LongestSearcher searcher = trie.getLongestSearcher(text.toCharArray(), 0);
        DoubleArrayTrie<String>.LongestSearcher searcherValueNull = trieValueNull.getLongestSearcher(text.toCharArray(), 0);

        while (true) {
            boolean next = searcher.next();
            boolean nextValueNull = searcherValueNull.next();

            if (next && nextValueNull) {
                assertTrue(searcher.begin == searcherValueNull.begin && searcher.length == searcherValueNull.length);
            } else if (next || nextValueNull) {
                assert false;
                break;
            } else {
                break;
            }
        }
    }

    public void testTransmit() throws Exception
    {
        DoubleArrayTrie<CoreDictionary.Attribute> dat = CustomDictionary.DEFAULT.dat;
        int index = dat.transition("钱", 1);
        assertNull(dat.output(index));
        index = dat.transition("龙", index);
        assertEquals("n 1 ", dat.output(index).toString());
    }

//    public void testCombine() throws Exception
//    {
//        DoubleArrayTrie<CoreDictionary.Attribute> dat = CustomDictionary.dat;
//        String[] wordNet = new String[]
//            {
//                "他",
//                "一",
//                "丁",
//                "不",
//                "识",
//                "一",
//                "丁",
//                "呀",
//            };
//        for (int i = 0; i < wordNet.length; ++i)
//        {
//            int state = 1;
//            state = dat.transition(wordNet[i], state);
//            if (state > 0)
//            {
//                int start = i;
//                int to = i + 1;
//                int end = - 1;
//                CoreDictionary.Attribute value = null;
//                for (; to < wordNet.length; ++to)
//                {
//                    state = dat.transition(wordNet[to], state);
//                    if (state < 0) break;
//                    CoreDictionary.Attribute output = dat.output(state);
//                    if (output != null)
//                    {
//                        value = output;
//                        end = to + 1;
//                    }
//                }
//                if (value != null)
//                {
//                    StringBuilder sbTerm = new StringBuilder();
//                    for (int j = start; j < end; ++j)
//                    {
//                        sbTerm.append(wordNet[j]);
//                    }
//                    System.out.println(sbTerm.toString() + "/" + value);
//                    i = end - 1;
//                }
//            }
//        }
//    }

    public void testHandleEmptyString() throws Exception
    {
        String emptyString = "";
        DoubleArrayTrie<String> dat = new DoubleArrayTrie<String>();
        TreeMap<String, String> dictionary = new TreeMap<String, String>();
        dictionary.put("bug", "问题");
        dat.build(dictionary);
        DoubleArrayTrie<String>.Searcher searcher = dat.getSearcher(emptyString, 0);
        while (searcher.next())
        {
        }
    }

    public void testIssue966() throws Exception
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        for (String word : "001乡道, 北京, 北京市通信公司, 来广营乡, 通州区".split(", "))
        {
            map.put(word, word);
        }
        DoubleArrayTrie<String> trie = new DoubleArrayTrie<String>(map);
        DoubleArrayTrie<String>.LongestSearcher searcher = trie.getLongestSearcher("北京市通州区001乡道发生了一件有意思的事情，来广营乡歌舞队正在跳舞", 0);
        while (searcher.next())
        {
            System.out.printf("%d %s\n", searcher.begin, searcher.value);
        }
    }

    public void testEnableFastBuild() {
        TreeMap<String, String> map = new TreeMap<String, String>();
        IOUtil.LineIterator iterator = new IOUtil.LineIterator("data/dictionary/CoreNatureDictionary.mini.txt");
        while (iterator.hasNext())
        {
            String line = iterator.next();
            map.put(line, line);
        }

        long startTimeMillis1 = System.currentTimeMillis();
        DoubleArrayTrie<String> trie1 = new DoubleArrayTrie<String>();
        trie1.build(map);
        long costTimeMillis1 = System.currentTimeMillis() - startTimeMillis1;

        long startTimeMillis2 = System.currentTimeMillis();
        DoubleArrayTrie<String> trie2 = new DoubleArrayTrie<String>(true);
        trie2.build(map);
        long costTimeMillis2 = System.currentTimeMillis() - startTimeMillis2;

        System.out.printf("[trie1]size=%s,costTimeMillis=%s\n", trie1.size, costTimeMillis1);
        System.out.printf("[trie2]size=%s,costTimeMillis=%s\n", trie2.size, costTimeMillis2);
        Assert.assertTrue(trie1.size < trie2.size && trie2.size < trie1.size * 1.5);
        Assert.assertTrue(costTimeMillis2 < costTimeMillis1 / 1.5);
    }
}