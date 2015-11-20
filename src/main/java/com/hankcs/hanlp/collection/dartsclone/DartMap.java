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

import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.corpus.io.ByteArray;

import java.io.DataOutputStream;
import java.util.*;

/**
 * 双数组trie树map，更省内存，原本希望代替DoubleArrayTrie，后来发现效率不够
 * @author hankcs
 */
public class DartMap<V> extends DoubleArray implements Map<String, V>, ITrie<V>
{
    V[] valueArray;

    public DartMap(List<String> keyList, V[] valueArray)
    {
        int[] indexArray = new int[valueArray.length];
        for (int i = 0; i < indexArray.length; ++i)
        {
            indexArray[i] = i;
        }
        this.valueArray = valueArray;
        build(keyList, indexArray);
    }

    public DartMap(TreeMap<String, V> map)
    {
        build(map);
    }

    public DartMap()
    {
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

    @Override
    public int build(TreeMap<String, V> keyValueMap)
    {
        int size = keyValueMap.size();
        int[] indexArray = new int[size];
        valueArray = (V[]) keyValueMap.values().toArray();
        List<String> keyList = new ArrayList<String>(size);
        int i = 0;
        for (Entry<String, V> entry : keyValueMap.entrySet())
        {
            indexArray[i] = i;
            valueArray[i] = entry.getValue();
            keyList.add(entry.getKey());
            ++i;
        }
        build(keyList, indexArray);
        return 0;
    }

    @Override
    public boolean save(DataOutputStream out)
    {
        return false;
    }

    @Override
    public boolean load(ByteArray byteArray, V[] value)
    {
        return false;
    }

    @Override
    public V get(char[] key)
    {
        return get(new String(key));
    }

    public V get(String key)
    {
        int id = exactMatchSearch(key);
        if (id == -1) return null;
        return valueArray[id];
    }

    @Override
    public V[] getValueArray(V[] a)
    {
        return valueArray;
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
            resultList.add(new Pair<String, V>(new String(keyBytes, 0, pair.first), valueArray[pair.second]));
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
        return Arrays.asList(valueArray);
    }

    @Override
    public Set<Entry<String, V>> entrySet()
    {
        throw new UnsupportedOperationException("双数组不支持");
    }
}
