/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/11 0:53</create-date>
 *
 * <copyright file="ViterbiEx.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algoritm;

import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.dictionary.TransformMatrixDictionary;

import java.util.*;

/**
 * 优化的维特比算法
 * @author hankcs
 */
public class ViterbiEx<E extends Enum<E>>
{
    List<EnumItem<E>> roleTagList;
    TransformMatrixDictionary<E> transformMatrixDictionary;

    public ViterbiEx(List<EnumItem<E>> roleTagList, TransformMatrixDictionary<E> transformMatrixDictionary)
    {
        this.roleTagList = roleTagList;
        this.transformMatrixDictionary = transformMatrixDictionary;
    }

    public List<E> computeTagList()
    {
        int length = roleTagList.size() - 1;
        List<E> tagList = new LinkedList<E>();
        double[][] cost = new double[length][];
        Iterator<EnumItem<E>> iterator = roleTagList.iterator();
        EnumItem<E> start = iterator.next();
        E pre = start.labelMap.entrySet().iterator().next().getKey();
        // 第一个是确定的
        tagList.add(pre);
        double total = 0.0;
        for (int i = 0; i < cost.length; ++i)
        {
            EnumItem<E> item = iterator.next();
            cost[i] = new double[item.labelMap.size()];
            Map.Entry<E, Integer>[] entryArray = new Map.Entry[item.labelMap.size()];
            Set<Map.Entry<E, Integer>> entrySet = item.labelMap.entrySet();
            Iterator<Map.Entry<E, Integer>> _i = entrySet.iterator();
            for (int _ = 0; _ < entryArray.length; ++_)
            {
                entryArray[_] = _i.next();
            }
            for (int j = 0; j < cost[i].length; ++j)
            {
                E cur = entryArray[j].getKey();
                cost[i][j] = total + transformMatrixDictionary.transititon_probability[pre.ordinal()][cur.ordinal()] - Math.log((item.getFrequency(cur) + 1e-8) / transformMatrixDictionary.getTotalFrequency(cur));
            }
            double perfect_cost_line = Double.MAX_VALUE;
            int perfect_j = 0;
            for (int j = 0; j < cost[i].length; ++j)
            {
                if (perfect_cost_line > cost[i][j])
                {
                    perfect_cost_line = cost[i][j];
                    perfect_j = j;
                }
            }
            total = perfect_cost_line;
            pre = entryArray[perfect_j].getKey();
            tagList.add(pre);
        }
//        if (HanLP.Config.DEBUG)
//        {
//            System.out.printf("viterbi_weight:%f\n", total);
//        }
        return tagList;
    }
}
