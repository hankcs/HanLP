package com.hankcs.hanlp.model.crf;

import junit.framework.TestCase;

public class LogLinearModelTest extends TestCase
{
    public void testLoad() throws Exception
    {
        LogLinearModel model = new LogLinearModel("/Users/hankcs/Downloads/crfpp-msr-cws-model.txt");
    }
}