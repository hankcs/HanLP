/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/15 10:23</create-date>
 *
 * <copyright file="ValueArray.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;

/**
 * 对值数组的包装，可以方便地取下一个
 * @author hankcs
 */
public class _ValueArray<V>
{
    V[] value;
    int offset;

    public _ValueArray(V[] value)
    {
        this.value = value;
    }

    public V nextValue()
    {
        return value[offset++];
    }

    /**
     * 仅仅给子类用，不要用
     */
    protected _ValueArray()
    {
    }

    public _ValueArray setValue(V[] value)
    {
        this.value = value;
        return this;
    }
}
