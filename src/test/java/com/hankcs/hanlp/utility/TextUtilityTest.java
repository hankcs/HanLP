package com.hankcs.hanlp.utility;

import junit.framework.TestCase;

public class TextUtilityTest extends TestCase
{
    public void testIsAllSingleByte() throws Exception
    {
        assertEquals(false, TextUtility.isAllSingleByte("中文a"));
        assertEquals(true, TextUtility.isAllSingleByte("abcABC!@#"));
    }
}