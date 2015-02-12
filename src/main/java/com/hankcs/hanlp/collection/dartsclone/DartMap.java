/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/22 18:17</create-date>
 *
 * <copyright file="DartMap.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.dartsclone;

import java.util.*;

/**
 * 双数组trie树map，更省内存，原本希望代替DoubleArrayTrie，后来发现效率不够
 * @author hankcs
 */
public class DartMap<V> extends DoubleArray implements Map<String, V>
{
    ArrayList<V> valueArray;

    public DartMap(List<String> keyList, List<V> valueList)
    {
        int[] valueArray = new int[valueList.size()];
        for (int i = 0; i < valueArray.length; ++i)
        {
            valueArray[i] = i;
        }
        this.valueArray = new ArrayList<V>(valueList);
        build(keyList, valueArray);
    }

    @Override
    public boolean isEmpty()
    {
        return size() == 0;
    }

    @Override
    public boolean containsKey(Object key)
    {
        return containsKey(key.toString());
    }

    /**
     * 是否包含key
     *
     * @param key
     * @return
     */
    public boolean containsKey(String key)
    {
        return exactMatchSearch(key) != -1;
    }

    @Override
    public boolean containsValue(Object value)
    {
        return false;
    }

    @Override
    public V get(Object key)
    {
        return get(key.toString());
    }

    public V get(String key)
    {
        int id = exactMatchSearch(key);
        if (id == -1) return null;
        return valueArray.get(id);
    }

    /**
     * 前缀查询
     * @param key
     * @param offset
     * @param maxResults
     * @return
     */
    public ArrayList<Pair<String, V>> commonPrefixSearch(String key, int offset, int maxResults)
    {
        byte[] keyBytes = key.getBytes(utf8);
        List<Pair<Integer, Integer>> pairList = commonPrefixSearch(keyBytes, offset, maxResults);
        ArrayList<Pair<String, V>> resultList = new ArrayList<Pair<String, V>>(pairList.size());
        for (Pair<Integer, Integer> pair : pairList)
        {
            resultList.add(new Pair<String, V>(new String(keyBytes, 0, pair.first), valueArray.get(pair.second)));
        }
        return resultList;
    }

    public ArrayList<Pair<String, V>> commonPrefixSearch(String key)
    {
        return commonPrefixSearch(key, 0, Integer.MAX_VALUE);
    }

    @Override
    public V put(String key, V value)
    {
        throw new UnsupportedOperationException("双数组不支持增量式插入");
    }

    @Override
    public V remove(Object key)
    {
        throw new UnsupportedOperationException("双数组不支持删除");
    }

    @Override
    public void putAll(Map<? extends String, ? extends V> m)
    {
        throw new UnsupportedOperationException("双数组不支持增量式插入");
    }

    @Override
    public void clear()
    {
        throw new UnsupportedOperationException("双数组不支持");
    }

    @Override
    public Set<String> keySet()
    {
        throw new UnsupportedOperationException("双数组不支持");
    }

    @Override
    public Collection<V> values()
    {
        return valueArray;
    }

    @Override
    public Set<Entry<String, V>> entrySet()
    {
        throw new UnsupportedOperationException("双数组不支持");
    }
}
