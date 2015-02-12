/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/17 9:47</create-date>
 *
 * <copyright file="BinSearch.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algoritm;

import java.util.Arrays;
import java.util.TreeSet;

/**
 * 求两个集合中最相近的两个数
 *
 * @author hankcs
 */
public class BinarySearch
{
    public static void main(String[] args)
    {
        TreeSet<Long> setA = new TreeSet<Long>();
        setA.add(5L);
        setA.add(4L);

        TreeSet<Long> setB = new TreeSet<Long>();
        setB.add(1L);
        setB.add(2L);
        setB.add(3L);
        setB.add(8L);
        setB.add(16L);
        System.out.println(computeMinimumDistance(setA, setB));
    }

    public static Long computeMinimumDistance(TreeSet<Long> setA, TreeSet<Long> setB)
    {
        Long[] arrayA = setA.toArray(new Long[0]);
        Long[] arrayB = setB.toArray(new Long[0]);
       return computeMinimumDistance(arrayA, arrayB);
    }

    public static Long computeMinimumDistance(Long[] arrayA, Long[] arrayB)
    {
        int startIndex = 0;
        Long min_distance = Long.MAX_VALUE;
        for (int i = 0; i < arrayA.length; ++i)
        {
            startIndex = Arrays.binarySearch(arrayB, startIndex, arrayB.length, arrayA[i]);
            if (startIndex < 0)
            {
                startIndex = -startIndex - 1;
                if (startIndex - 1 >= 0)
                {
                    min_distance = Math.min(min_distance, arrayA[i] - arrayB[startIndex - 1]);
                }
                if (startIndex < arrayB.length)
                {
                    min_distance = Math.min(min_distance, arrayB[startIndex] - arrayA[i]);
                }
            }
            else return 0L;
        }

        return min_distance;
    }

    public static Long computeAverageDistance(Long[] arrayA, Long[] arrayB)
    {
        Long totalA = 0L;
        Long totalB = 0L;
        for (Long a : arrayA) totalA += a;
        for (Long b : arrayB) totalB += b;

        return Math.abs(totalA / arrayA.length - totalB / arrayB.length);
    }
}
