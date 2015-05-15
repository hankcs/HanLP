/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/15 10:24</create-date>
 *
 * <copyright file="EmptyValueArray.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;

/**
 * @author hankcs
 */
public class _EmptyValueArray<V> extends _ValueArray<V>
{
    public _EmptyValueArray()
    {
    }

    @Override
    public V nextValue()
    {
        return null;
    }
}
