/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/2 17:32</create-date>
 *
 * <copyright file="TestAddressRecognition.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.NShort.NShortSegment;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestAddressRecognition extends TestCase
{
    public static void main(String[] args)
    {
        System.out.println(NShortSegment.parse("地址：乌鲁木齐南路218、228"));
    }
}
