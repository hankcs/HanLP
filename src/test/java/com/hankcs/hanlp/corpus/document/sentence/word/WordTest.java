package com.hankcs.hanlp.corpus.document.sentence.word;

import junit.framework.TestCase;

public class WordTest extends TestCase
{
    public void testCreate() throws Exception
    {
        assertEquals("人民网/nz", Word.create("人民网/nz").toString());
        assertEquals("[纽约/nsf 时报/n]/nz", CompoundWord.create("[纽约/nsf 时报/n]/nz").toString());
        assertEquals("[中央/n 人民/n 广播/vn 电台/n]/nt", CompoundWord.create("[中央/n 人民/n 广播/vn 电台/n]nt").toString());
    }

    public void testSpace() throws Exception
    {
        CompoundWord compoundWord = CompoundWord.create("[9/m  11/m 后/f]/mq");
        assertEquals(3, compoundWord.innerList.size());
    }
}