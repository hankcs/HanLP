/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 11:06</create-date>
 *
 * <copyright file="CoNllLine.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dependency.CoNll;

/**
 * CoNLL语料中的一行
 * @author hankcs
 */
public class CoNllLine
{
    /**
     * 十个值
     */
    public String[] value = new String[10];

    /**
     * 第一个值化为id
     */
    public int id;

    public CoNllLine(String... args)
    {
        int length = Math.min(args.length, value.length);
        for (int i = 0; i < length; ++i)
        {
            value[i] = args[i];
        }
        id = Integer.parseInt(value[0]);
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder();
        for (String value : this.value)
        {
            sb.append(value);
            sb.append('\t');
        }
        return sb.deleteCharAt(sb.length() - 1).toString();
    }
}
