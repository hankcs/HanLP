package com.hankcs.hanlp.corpus.tag;

import junit.framework.TestCase;

public class NatureTest extends TestCase
{
    public void testFromString() throws Exception
    {
        Nature one = Nature.create("新词性1");
        Nature two = Nature.create("新词性2");

        assertEquals(one, Nature.fromString("新词性1"));
        assertEquals(two, Nature.fromString("新词性2"));
    }
}