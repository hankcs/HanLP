/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/5/14 19:20</create-date>
 *
 * <copyright file="ChainItem.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.dynamicarray;

/**
 * @author He Han
 */
public class ChainItem<T>
{
    public int row;
    public int col;
    public T Content;
    public ChainItem<T> next;

    public void copy(ChainItem<T> other)
    {
        row = other.row;
        col = other.col;
        Content = other.Content;
        next = other.next;
    }
}
