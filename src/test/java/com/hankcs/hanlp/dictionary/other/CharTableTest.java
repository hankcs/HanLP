package com.hankcs.hanlp.dictionary.other;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class CharTableTest extends TestCase
{
    public void testConvert() throws Exception
    {
        HanLP.Config.enableDebug();
        assertEquals('a', CharTable.convert('A'));
    }
}