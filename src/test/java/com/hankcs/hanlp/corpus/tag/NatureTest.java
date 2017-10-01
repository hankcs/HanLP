package com.hankcs.hanlp.corpus.tag;

import com.hankcs.hanlp.corpus.util.CustomNatureUtility;
import junit.framework.TestCase;

public class NatureTest extends TestCase
{
    public void testFromString() throws Exception
    {
        Nature one = CustomNatureUtility.addNature("新词性1");
        Nature two = CustomNatureUtility.addNature("新词性2");

        assertEquals(one, Nature.fromString("新词性1"));
        assertEquals(two, Nature.fromString("新词性2"));
    }
}