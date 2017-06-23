/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-07 AM11:49</create-date>
 *
 * <copyright file="GlobalObjectPool.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Map;

/**
 * 全局对象缓存池<br>
 * 用于储存那些体积庞当的模型，如果该模型已经被加载过一次，那么就不需要重新加载。同时，如果JVM内存不够，并且没有任何强引用时，允许垃圾
 * 回收器回收这些模型。
 *
 * @author hankcs
 */
@SuppressWarnings("unchecked")
public class GlobalObjectPool
{
    /**
     * 缓存池
     */
    private static Map<Object, SoftReference> pool = new HashMap<Object, SoftReference>();

    /**
     * 获取对象
     * @param id 对象的id，可以是任何全局唯一的标示符
     * @param <T> 对象类型
     * @return 对象
     */
    public synchronized static <T> T get(Object id)
    {
        SoftReference reference = pool.get(id);
        if (reference == null) return null;
        return (T) reference.get();
    }

    /**
     * 存放全局变量
     * @param id
     * @param <T>
     * @return
     */
    public synchronized static <T> T put(Object id, T value)
    {
        SoftReference old = pool.put(id, new SoftReference(value));
        return old == null ? null : (T) old.get();
    }

    /**
     * 清空全局变量
     */
    public synchronized static void clear()
    {
        pool.clear();
    }
}