/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/7 9:59</create-date>
 *
 * <copyright file="TestPinyinKey.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.algorithm.LongestCommonSubstring;
import com.hankcs.hanlp.suggest.scorer.pinyin.PinyinKey;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestPinyinKey extends TestCase
{
    public void testConstruct() throws Exception
    {
        PinyinKey pinyinKeyA = new PinyinKey("专题分析");
        PinyinKey pinyinKeyB = new PinyinKey("教室资格");
        System.out.println(pinyinKeyA);
        System.out.println(pinyinKeyB);
        System.out.println(LongestCommonSubstring.compute(pinyinKeyA.getFirstCharArray(), pinyinKeyB.getFirstCharArray()));
    }
}
