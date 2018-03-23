package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

public class PerceptionNERecognizerTest extends TestCase
{
    public void testEmptyInput() throws Exception
    {
        PerceptionNERecognizer recognizer = new PerceptionNERecognizer();
        recognizer.recognize(new String[0], new String[0]);
    }
}