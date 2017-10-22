package com.hankcs.hanlp.dictionary;

import junit.framework.TestCase;

public class CoreBiGramTableDictionaryTest extends TestCase
{
    public void testReload() throws Exception
    {
        int biFrequency = CoreBiGramTableDictionary.getBiFrequency("高性能", "计算");
        CoreBiGramTableDictionary.reload();
        assertEquals(biFrequency, CoreBiGramTableDictionary.getBiFrequency("高性能", "计算"));
    }
}