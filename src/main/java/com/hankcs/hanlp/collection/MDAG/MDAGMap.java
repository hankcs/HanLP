/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/21 18:59</create-date>
 *
 * <copyright file="MDAGMap.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.collection.MDAG;

import com.hankcs.hanlp.utility.ByteUtil;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class MDAGMap<V> extends AbstractMap<String, V>
{
    ArrayList<V> valueList = new ArrayList<>();
    MDAGForMap mdag = new MDAGForMap();

    @Override
    public V put(String key, V value)
    {
        V origin = get(key);
        if (origin == null)
        {
            valueList.add(value);
            char[] twoChar = ByteUtil.convertIntToTwoChar(valueList.size() - 1);
            mdag.addString(key + MDAGForMap.DELIMITER + twoChar[0] + twoChar[1]);
        }
        return origin;
    }

    @Override
    public V get(Object key)
    {
        int valueIndex = mdag.getValueIndex(key.toString());
        if (valueIndex != -1)
        {
            return valueList.get(valueIndex);
        }
        return null;
    }

    public V get(String key)
    {
        int valueIndex = mdag.getValueIndex(key);
        if (valueIndex != -1)
        {
            return valueList.get(valueIndex);
        }
        return null;
    }

    @Override
    public Set<Entry<String, V>> entrySet()
    {
        return null;
    }

    static class MDAGForMap extends MDAG
    {
        static final char DELIMITER = Character.MIN_VALUE;
        public int getValueIndex(String key)
        {
            key += DELIMITER;
            if (sourceNode != null)      //if the MDAG hasn't been simplified
            {
                MDAGNode targetNode = sourceNode.transition(key);
                if (targetNode == null) return -1;
                // 接下来应该是一条单链路
                TreeMap<Character, MDAGNode> outgoingTransitions = targetNode.getOutgoingTransitions();
                assert outgoingTransitions.size() == 1 : "不是单链！";
                char high = outgoingTransitions.keySet().iterator().next();
                targetNode = targetNode.transition(high);
                outgoingTransitions = targetNode.getOutgoingTransitions();
                assert outgoingTransitions.size() == 1 : "不是单链！";
                char low = outgoingTransitions.keySet().iterator().next();
                return ByteUtil.convertTwoCharToInt(high, low);
            }
            else
            {
                SimpleMDAGNode targetNode = simplifiedSourceNode.transition(mdagDataArray, key.toCharArray());
                if (targetNode == null) return -1;
            }

            return 0;
        }
    }
}
