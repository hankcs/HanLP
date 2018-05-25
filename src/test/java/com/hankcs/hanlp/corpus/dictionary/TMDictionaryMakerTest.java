package com.hankcs.hanlp.corpus.dictionary;

import junit.framework.TestCase;

public class TMDictionaryMakerTest extends TestCase
{
    public void testCreate() throws Exception
    {
        TMDictionaryMaker tmDictionaryMaker = new TMDictionaryMaker();
        tmDictionaryMaker.addPair("ab", "cd");
        tmDictionaryMaker.addPair("ab", "cd");
        tmDictionaryMaker.addPair("ab", "Y");
        tmDictionaryMaker.addPair("ef", "gh");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("X", "Y");
//        System.out.println(tmDictionaryMaker);
    }
}