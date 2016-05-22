/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/29 11:22</create-date>
 *
 * <copyright file="TestPersonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.dictionary.nr.PersonDictionary;

/**
 * @author hankcs
 */
public class TestPersonDictionary
{
    public static void main(String[] args)
    {
        System.out.println(PersonDictionary.dictionary.get("稽"));
    }
}
