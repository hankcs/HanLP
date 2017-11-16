/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-11-14 下午8:32</create-date>
 *
 * <copyright file="EnumItemDictionary.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.common;

import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.io.ByteArray;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;

/**
 * 枚举条目的通用词典（类似C++的模板，然而Java并没有模板机制，所以有几个方法需要子类啰嗦一下）
 *
 * @author hankcs
 */
public abstract class EnumItemDictionary<E extends Enum<E>> extends CommonDictionary<EnumItem<E>>
{
    @Override
    protected EnumItem<E> createValue(String[] params)
    {
        Map.Entry<String, Map.Entry<String, Integer>[]> args = EnumItem.create(params);
        EnumItem<E> nrEnumItem = new EnumItem<E>();
        for (Map.Entry<String, Integer> e : args.getValue())
        {
            nrEnumItem.labelMap.put(valueOf(e.getKey()), e.getValue());
        }
        return nrEnumItem;
    }

    /**
     * 代理E.valueOf
     *
     * @param name
     * @return
     */
    protected abstract E valueOf(String name);

    /**
     * 代理E.values
     *
     * @return
     */
    protected abstract E[] values();

    /**
     * 代理new EnumItem<E>
     *
     * @return
     */
    protected abstract EnumItem<E> newItem();

    @Override
    final protected EnumItem<E>[] loadValueArray(ByteArray byteArray)
    {
        if (byteArray == null)
        {
            return null;
        }
        E[] nrArray = values();
        int size = byteArray.nextInt();
        EnumItem<E>[] valueArray = new EnumItem[size];
        for (int i = 0; i < size; ++i)
        {
            int currentSize = byteArray.nextInt();
            EnumItem<E> item = newItem();
            for (int j = 0; j < currentSize; ++j)
            {
                E nr = nrArray[byteArray.nextInt()];
                int frequency = byteArray.nextInt();
                item.labelMap.put(nr, frequency);
            }
            valueArray[i] = item;
        }
        return valueArray;
    }

    @Override
    protected void saveValue(EnumItem<E> item, DataOutputStream out) throws IOException
    {
        out.writeInt(item.labelMap.size());
        for (Map.Entry<E, Integer> entry : item.labelMap.entrySet())
        {
            out.writeInt(entry.getKey().ordinal());
            out.writeInt(entry.getValue());
        }
    }
}
