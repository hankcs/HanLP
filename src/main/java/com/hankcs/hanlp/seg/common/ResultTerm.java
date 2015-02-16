/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/2/14 0:05</create-date>
 *
 * <copyright file="ResultTerm.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.common;

/**
 * 一个通用的Term
* @author hankcs
*/
public class ResultTerm<V>
{
    public String word;
    public V label;
    public int offset;

    public ResultTerm(String word, V label, int offset)
    {
        this.word = word;
        this.label = label;
        this.offset = offset;
    }

    @Override
    public String toString()
    {
        return word + '/' + label;
    }
}
