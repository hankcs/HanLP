package com.hankcs.hanlp.model.hmm;

import junit.framework.TestCase;

import java.util.Arrays;

public class HMMPOSTaggerTest extends TestCase
{
    public void testTrain() throws Exception
    {
        HMMPOSTagger tagger = new HMMPOSTagger();
        tagger.train("data/test/pku98/199801.txt");
        System.out.println(Arrays.toString(tagger.tag("我", "的", "希望", "是", "希望", "和平")));
    }
}