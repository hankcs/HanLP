package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.dictionary.other.CharType;
import junit.framework.TestCase;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TextUtilityTest extends TestCase
{
    public void testIsAllSingleByte() throws Exception
    {
        assertEquals(false, TextUtility.isAllSingleByte("中文a"));
        assertEquals(true, TextUtility.isAllSingleByte("abcABC!@#"));
    }

    @Test
    public void testChineseNum()
    {
        assertEquals(true, TextUtility.isAllChineseNum("两千五百万"));
        assertEquals(true, TextUtility.isAllChineseNum("两千分之一"));
        assertEquals(true, TextUtility.isAllChineseNum("几十"));
        assertEquals(true, TextUtility.isAllChineseNum("十几"));
        assertEquals(false,TextUtility.isAllChineseNum("上来"));
    }

    @Test
    public void testArabicNum()
    {
        assertEquals(true, TextUtility.isAllNum("2.5"));
        assertEquals(true, TextUtility.isAllNum("3600"));
        assertEquals(true, TextUtility.isAllNum("500万"));
        assertEquals(true, TextUtility.isAllNum("87.53%"));
        assertEquals(true, TextUtility.isAllNum("５５０"));
        assertEquals(true, TextUtility.isAllNum("１０％"));
        assertEquals(true, TextUtility.isAllNum("98．1％"));
        assertEquals(false, TextUtility.isAllNum("，"));
    }
}