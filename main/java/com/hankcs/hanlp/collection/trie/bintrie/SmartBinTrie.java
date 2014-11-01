/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 18:10</create-date>
 *
 * <copyright file="SmartBinTrie.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie.bintrie;

/**
 * @author hankcs
 */
public class SmartBinTrie<V> extends BinTrie<V>
{
    @Override
    protected BaseNode<V> newNode(char c, Status status, V value)
    {
        return new SmartNode<V>(c, status, value);
    }

    @Override
    protected BaseNode<V> newInstance()
    {
        return new SmartNode<V>();
    }
}
