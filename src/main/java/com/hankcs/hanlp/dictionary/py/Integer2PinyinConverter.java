/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/7 10:02</create-date>
 *
 * <copyright file="Pinyin2Integer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.py;

/**
 * 将整型转为拼音
 * @author hankcs
 */
public class Integer2PinyinConverter
{
    public static final Pinyin[] pinyins = Pinyin.values();

    public static Pinyin getPinyin(int ordinal)
    {
        return pinyins[ordinal];
    }
}
