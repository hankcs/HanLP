package com.hankcs.hanlp.collection.MDAG;

import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.util.Set;
import java.util.TreeSet;

public class MDAGMapTest extends TestCase
{
    MDAGMap<Integer> mdagMap = new MDAGMap<>();
    Set<String> validKeySet;

    public void setUp() throws Exception
    {
        IOUtil.LineIterator iterator = new IOUtil.LineIterator("data/dictionary/custom/CustomDictionary.txt");
        validKeySet = new TreeSet<>();
        while (iterator.hasNext())
        {
            validKeySet.add(iterator.next().split(" ")[0]);
        }
    }

    public void testPut() throws Exception
    {
        for (String word : validKeySet)
        {
            mdagMap.put(word, word.length());
        }
    }

    public void testGet() throws Exception
    {
        testPut();
        for (String word : validKeySet)
        {
            assertEquals(word.length(), (int)mdagMap.get(word));
        }
    }
}