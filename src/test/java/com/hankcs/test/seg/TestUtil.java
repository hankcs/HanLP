/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 22:26</create-date>
 *
 * <copyright file="TestUtil.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.utility.TextUtility;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestUtil extends TestCase
{
    public void testNonZero() throws Exception
    {
        System.out.println(TextUtility.nonZero(0));
    }
}
