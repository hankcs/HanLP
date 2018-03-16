package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

public class PerceptronSegmenterTest extends TestCase
{
    public void testEmptyString() throws Exception
    {
        PerceptronSegmenter segmenter = new PerceptronSegmenter();
        segmenter.segment("");
    }
}