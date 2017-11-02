/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/13 PM3:48</create-date>
 *
 * <copyright file="FrequencyMap.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.collections;

import java.util.TreeMap;

/**
 * 统计词频的Map
 * @author hankcs
 */
public class FrequencyMap<K> extends TreeMap<K, int[]>
{
    /**
     * 增加一个词的词频
     * @param key
     * @return
     */
    public int add(K key)
    {
        int[] f = get(key);
        if (f == null)
        {
            f = new int[]{1};
            put(key, f);
        }
        else ++f[0];

        return f[0];
    }
}