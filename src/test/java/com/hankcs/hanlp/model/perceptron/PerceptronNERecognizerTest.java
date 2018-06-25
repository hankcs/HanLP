package com.hankcs.hanlp.model.perceptron;

import junit.framework.TestCase;

public class PerceptronNERecognizerTest extends TestCase
{
    public void testEmptyInput() throws Exception
    {
        PerceptronNERecognizer recognizer = new PerceptronNERecognizer();
        recognizer.recognize(new String[0], new String[0]);
    }
}