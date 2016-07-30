/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/7 19:02</create-date>
 *
 * <copyright file="DemoSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.other.CharTable;

/**
 * 简繁转换
 * @author hankcs
 */
public class DemoTraditionalChinese2SimplifiedChinese
{
    public static void main(String[] args)
    {
        System.out.println(HanLP.convertToTraditionalChinese("“以后等你当上皇后，就能买草莓庆祝了”。发现一根白头发"));
        System.out.println(HanLP.convertToSimplifiedChinese("憑藉筆記簿型電腦寫程式HelloWorld"));
    }
}
