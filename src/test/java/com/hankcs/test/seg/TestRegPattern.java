/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/20 18:01</create-date>
 *
 * <copyright file="testRegPattern.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import junit.framework.TestCase;

import java.util.regex.Pattern;

/**
 * 测试正则匹配数词
 * @author hankcs
 */
public class TestRegPattern extends TestCase
{
    public void testPattern()
    {
        assertEquals(true, Pattern.compile("^(-?\\d+)(\\.\\d+)?$").matcher("2014").matches());  // 浮点数
        assertEquals(true, Pattern.compile("^(-?\\d+)(\\.\\d+)?$").matcher("-2014").matches());  // 浮点数
        assertEquals(true, Pattern.compile("^(-?\\d+)(\\.\\d+)?$").matcher("20.14").matches());  // 浮点数
    }
}
