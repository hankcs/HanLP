/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/28 21:50</create-date>
 *
 * <copyright file="PersonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

/**
 * 人名字典
 * @author hankcs
 */
public class ZZPersonDictionary
{
    static UnknownWordDictionary unknownWordDictionary;
    static ContextStat contextStat;
    static
    {
        unknownWordDictionary = new UnknownWordDictionary("人名");
        unknownWordDictionary.load("data/dictionary/person/nr.txt");
        contextStat = new ContextStat();
        contextStat.Load("data/dictionary/person/nr.ctx.txt");
    }

    static public int getFrequency(String key, int pos)
    {
        return unknownWordDictionary.getFrequency(key, pos);
    }

    public static UnknownWordDictionary getUnknownWordDictionary()
    {
        return unknownWordDictionary;
    }

    public static ContextStat getContextStat()
    {
        return contextStat;
    }
}
