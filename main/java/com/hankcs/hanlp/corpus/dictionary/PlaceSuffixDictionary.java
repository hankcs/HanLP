/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/17 17:43</create-date>
 *
 * <copyright file="PlaceSuffixDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;

import com.hankcs.hanlp.corpus.dictionary.SuffixDictionary;
import com.hankcs.hanlp.utility.Predefine;

/**
 * 做一个简单的封装
 * @author hankcs
 */
public class PlaceSuffixDictionary
{
    public static SuffixDictionary dictionary = new SuffixDictionary();
    static
    {
        dictionary.addAll(Predefine.POSTFIX_SINGLE);
        dictionary.addAll(Predefine.POSTFIX_MUTIPLE);
    }
}
