/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/2/17 16:50</create-date>
 *
 * <copyright file="BiAutomataBin.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algoritm.automata.bin;

import com.hankcs.hanlp.algoritm.automata.IBiAutomata;
import com.hankcs.hanlp.corpus.io.ByteArray;

import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * 基于二分的简易自动机（施工中）
 * @author hankcs
 */
public class BiAutomataBin implements IBiAutomata
{
    int[][] matrix;

    /**
     * 构建自动机
     * @param map 表示状态的稀疏矩阵，状态id必须为正
     */
    public BiAutomataBin build(TreeMap<Integer, TreeSet<Integer>> map)
    {
        int maxValue = 0;
        // 找所有状态中的最大值
        for (Map.Entry<Integer, TreeSet<Integer>> entry : map.entrySet())
        {
            maxValue = Math.max(entry.getKey(), maxValue);
            for (Integer v : entry.getValue())
            {
                maxValue = Math.max(v, maxValue);
            }
        }
        matrix = new int[maxValue + 1][];
        // 填充值
        for (Map.Entry<Integer, TreeSet<Integer>> entry : map.entrySet())
        {
            Integer from = entry.getKey();
            TreeSet<Integer> toSet = entry.getValue();
            matrix[from] = new int[toSet.size()];
            int i = 0;
            for (Integer to : toSet)
            {
                matrix[from][i++] = to;
            }
        }

        return this;
    }

    /**
     * 转移状态
     * @param from 当前状态
     * @param to 目标状态
     * @return 如果可以转移则返回to，否则返回负数
     */
    private int _transmit(int from, int to)
    {
        if (from < 0 || from >= matrix.length) return -1;
        if (matrix[from] == null) return -1;
        if (Arrays.binarySearch(matrix[from], to) >= 0) return to;
        return -1;
    }

    public void save(DataOutputStream out) throws Exception
    {
        out.writeInt(matrix.length);
        for (int[] line : matrix)
        {
            if (line == null)
            {
                out.writeInt(-1);
            }
            else
            {
                out.writeInt(line.length);
                for (int v : line)
                {
                    out.writeInt(v);
                }
            }
        }
    }

    public BiAutomataBin load(ByteArray byteArray)
    {
        matrix = new int[byteArray.nextInt()][];
        for (int i = 0; i < matrix.length; i++)
        {
            int length = byteArray.nextInt();
            if (length > 0)
            {
                matrix[i] = new int[length];
                for (int j = 0; j < length; ++j)
                {
                    matrix[i][j] = byteArray.nextInt();
                }
            }
        }

        return this;
    }

    @Override
    public boolean transmit(int from, int to)
    {
        return uniTransmit(from, to);
//        return biTransmit(from, to);
    }

    /**
     * 二元转换
     * @param from
     * @param to
     * @return
     */
    private  boolean biTransmit(int from, int to)
    {
        if (from < 0 || from >= matrix.length) return false;
        if (matrix[from] == null) return false;
        if (Arrays.binarySearch(matrix[from], to) >= 0) return true;
        return false;
    }

    /**
     * 一元转换
     * @param from
     * @param to
     * @return
     */
    private  boolean uniTransmit(int from, int to)
    {
        if (from < 0 || from >= matrix.length) return false;
        return matrix[from] != null;
    }
}
