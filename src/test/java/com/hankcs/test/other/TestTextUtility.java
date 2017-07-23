package com.hankcs.test.other;

import static org.junit.Assert.*;

import org.junit.Test;

import com.hankcs.hanlp.utility.TextUtility;

public class TestTextUtility
{
    
    @Test
    public void testChineseNum()
    {
        assertEquals(Boolean.TRUE, TextUtility.isAllChineseNum("两千五百万"));
        assertEquals(Boolean.TRUE, TextUtility.isAllChineseNum("两千分之一"));
        assertEquals(Boolean.TRUE, TextUtility.isAllChineseNum("几十"));
        assertEquals(Boolean.TRUE, TextUtility.isAllChineseNum("十几"));
        assertEquals(Boolean.FALSE,TextUtility.isAllChineseNum("上来"));
    }

    @Test
    public void testArabicNum()
    {
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("2.5"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("3600"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("500万"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("87.53%"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("５５０"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("１０％"));
        assertEquals(Boolean.TRUE, TextUtility.isAllNum("98．1％"));
    }
    
}
