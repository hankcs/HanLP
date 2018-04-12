package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

import java.util.List;

public class PerceptronSegmenterTest extends TestCase
{

    private PerceptronSegmenter segmenter;

    @Override
    public void setUp() throws Exception
    {
        segmenter = new PerceptronSegmenter();
    }

    public void testEmptyString() throws Exception
    {
        segmenter.segment("");
    }

    public void testNRF() throws Exception
    {
        String text = "他们确保了唐纳德·特朗普在总统大选中获胜。";
        List<String> wordList = segmenter.segment(text);
        assertTrue(wordList.contains("唐纳德·特朗普"));
    }
}