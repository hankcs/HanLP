/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/4/23 0:23</create-date>
 *
 * <copyright file="ITrie.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2015, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.trie;

import com.hankcs.hanlp.corpus.io.ByteArray;

import java.io.DataOutputStream;
import java.util.TreeMap;

/**
 * trie树接口
 * @author hankcs
 */
public interface ITrie<V>
{
    int build(TreeMap<String, V> keyValueMap);
    boolean save(DataOutputStream out);
    boolean load(ByteArray byteArray, V[] value);
    V get(char[] key);
    V get(String key);
    V[] getValueArray(V[] a);
    boolean containsKey(String key);
    int size();
}
