/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/25 16:00</create-date>
 *
 * <copyright file="testSplitMiddleSlashFromDigitalWords.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.NShort.NShortSegment;

/**
 * @author hankcs
 */
public class TestSplitMiddleSlashFromDigitalWords
{
    public static void main(String[] args)
    {
//        System.out.println(Arrays.toString("sd-j".split("-")));
        System.out.println(NShortSegment.parse("3-4月份"));
    }
}
