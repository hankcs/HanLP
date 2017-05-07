/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:00</create-date>
 *
 * <copyright file="SimpleItem.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary.item;

import java.util.*;

/**
 * @author hankcs
 */
public class SimpleItem
{
    /**
     * 该条目的标签
     */
    public Map<String, Integer> labelMap;

    public SimpleItem()
    {
        labelMap = new TreeMap<String, Integer>();
    }

    public void addLabel(String label)
    {
        Integer frequency = labelMap.get(label);
        if (frequency == null)
        {
            frequency = 1;
        }
        else
        {
            ++frequency;
        }

        labelMap.put(label, frequency);
    }

    /**
     * 添加一个标签和频次
     * @param label
     * @param frequency
     */
    public void addLabel(String label, Integer frequency)
    {
        Integer innerFrequency = labelMap.get(label);
        if (innerFrequency == null)
        {
            innerFrequency = frequency;
        }
        else
        {
            innerFrequency += frequency;
        }

        labelMap.put(label, innerFrequency);
    }

    /**
     * 删除一个标签
     * @param label 标签
     */
    public void removeLabel(String label)
    {
        labelMap.remove(label);
    }

    public boolean containsLabel(String label)
    {
        return labelMap.containsKey(label);
    }

    public int getFrequency(String label)
    {
        Integer frequency = labelMap.get(label);
        if (frequency == null) return 0;
        return frequency;
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder();
        ArrayList<Map.Entry<String, Integer>> entries = new ArrayList<Map.Entry<String, Integer>>(labelMap.entrySet());
        Collections.sort(entries, new Comparator<Map.Entry<String, Integer>>()
        {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2)
            {
                return -o1.getValue().compareTo(o2.getValue());
            }
        });
        for (Map.Entry<String, Integer> entry : entries)
        {
            sb.append(entry.getKey());
            sb.append(' ');
            sb.append(entry.getValue());
            sb.append(' ');
        }
        return sb.toString();
    }

    public static SimpleItem create(String param)
    {
        if (param == null) return null;
        String[] array = param.split(" ");
        return create(array);
    }

    public static SimpleItem create(String param[])
    {
        if (param.length % 2 == 1) return null;
        SimpleItem item = new SimpleItem();
        int natureCount = (param.length) / 2;
        for (int i = 0; i < natureCount; ++i)
        {
            item.labelMap.put(param[2 * i], Integer.parseInt(param[1 + 2 * i]));
        }
        return item;
    }

    /**
     * 合并两个条目，两者的标签map会合并
     * @param other
     */
    public void combine(SimpleItem other)
    {
        for (Map.Entry<String, Integer> entry : other.labelMap.entrySet())
        {
            addLabel(entry.getKey(), entry.getValue());
        }
    }

    /**
     * 获取全部频次
     * @return
     */
    public int getTotalFrequency()
    {
        int frequency = 0;
        for (Integer f : labelMap.values())
        {
            frequency += f;
        }
        return frequency;
    }

    public String getMostLikelyLabel()
    {
        return labelMap.entrySet().iterator().next().getKey();
    }
}
