package com.hankcs.hanlp.model.perceptron.utility;

import junit.framework.TestCase;

import java.util.Arrays;

public class IOUtilityTest extends TestCase
{
    public void testReadLineToArray() throws Exception
    {
        String line = " 你好   世界 ! ";
        String[] array = IOUtility.readLineToArray(line);
        System.out.println(Arrays.toString(array));
    }
}