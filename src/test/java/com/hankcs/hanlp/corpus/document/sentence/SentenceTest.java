package com.hankcs.hanlp.corpus.document.sentence;

import junit.framework.TestCase;

public class SentenceTest extends TestCase
{
    public void testText() throws Exception
    {
        assertEquals("人民网纽约时报", Sentence.create("人民网/nz [纽约/nsf 时报/n]/nz").text());
    }
}