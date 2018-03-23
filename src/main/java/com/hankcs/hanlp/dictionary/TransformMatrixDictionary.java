/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:49</create-date>
 *
 * <copyright file="TransformMatrixDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.corpus.io.IOUtil;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 转移矩阵词典
 *
 * @param <E> 标签的枚举类型
 * @author hankcs
 */
public class TransformMatrixDictionary<E extends Enum<E>>
{
    Class<E> enumType;
    /**
     * 内部标签下标最大值不超过这个值，用于矩阵创建
     */
    private int ordinaryMax;

    public TransformMatrixDictionary(Class<E> enumType)
    {
        this.enumType = enumType;
    }

    /**
     * 储存转移矩阵
     */
    int matrix[][];

    /**
     * 储存每个标签出现的次数
     */
    int total[];

    /**
     * 所有标签出现的总次数
     */
    int totalFrequency;

    // HMM的五元组
    /**
     * 隐状态
     */
    public int[] states;
    //int[] observations;
    /**
     * 初始概率
     */
    public double[] start_probability;
    /**
     * 转移概率
     */
    public double[][] transititon_probability;

    public boolean load(String path)
    {
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(IOUtil.newInputStream(path), "UTF-8"));
            // 第一行是矩阵的各个类型
            String line = br.readLine();
            String[] _param = line.split(",");
            // 为了制表方便，第一个label是废物，所以要抹掉它
            String[] labels = new String[_param.length - 1];
            System.arraycopy(_param, 1, labels, 0, labels.length);
            int[] ordinaryArray = new int[labels.length];
            ordinaryMax = 0;
            for (int i = 0; i < ordinaryArray.length; ++i)
            {
                ordinaryArray[i] = convert(labels[i]).ordinal();
                ordinaryMax = Math.max(ordinaryMax, ordinaryArray[i]);
            }
            ++ordinaryMax;
            matrix = new int[ordinaryMax][ordinaryMax];
            for (int i = 0; i < ordinaryMax; ++i)
            {
                for (int j = 0; j < ordinaryMax; ++j)
                {
                    matrix[i][j] = 0;
                }
            }
            // 之后就描述了矩阵
            while ((line = br.readLine()) != null)
            {
                String[] paramArray = line.split(",");
                int currentOrdinary = convert(paramArray[0]).ordinal();
                for (int i = 0; i < ordinaryArray.length; ++i)
                {
                    matrix[currentOrdinary][ordinaryArray[i]] = Integer.valueOf(paramArray[1 + i]);
                }
            }
            br.close();
            // 需要统计一下每个标签出现的次数
            total = new int[ordinaryMax];
            for (int j = 0; j < ordinaryMax; ++j)
            {
                total[j] = 0;
                for (int i = 0; i < ordinaryMax; ++i)
                {
                    total[j] += matrix[j][i]; // 按行累加
                }
            }
            for (int j = 0; j < ordinaryMax; ++j)
            {
                if (total[j] == 0)
                {
                    for (int i = 0; i < ordinaryMax; ++i)
                    {
                        total[j] += matrix[i][j]; // 按列累加
                    }
                }
            }
            for (int j = 0; j < ordinaryMax; ++j)
            {
                totalFrequency += total[j];
            }
            // 下面计算HMM四元组
            states = ordinaryArray;
            start_probability = new double[ordinaryMax];
            for (int s : states)
            {
                double frequency = total[s] + 1e-8;
                start_probability[s] = -Math.log(frequency / totalFrequency);
            }
            transititon_probability = new double[ordinaryMax][ordinaryMax];
            for (int from : states)
            {
                for (int to : states)
                {
                    double frequency = matrix[from][to] + 1e-8;
                    transititon_probability[from][to] = -Math.log(frequency / total[from]);
//                    System.out.println("from" + NR.values()[from] + " to" + NR.values()[to] + " = " + transititon_probability[from][to]);
                }
            }
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "失败" + e);
            return false;
        }

        return true;
    }

    /**
     * 获取转移频次
     *
     * @param from
     * @param to
     * @return
     */
    public int getFrequency(String from, String to)
    {
        return getFrequency(convert(from), convert(to));
    }

    /**
     * 获取转移频次
     *
     * @param from
     * @param to
     * @return
     */
    public int getFrequency(E from, E to)
    {
        return matrix[from.ordinal()][to.ordinal()];
    }

    /**
     * 获取e的总频次
     *
     * @param e
     * @return
     */
    public int getTotalFrequency(E e)
    {
        return total[e.ordinal()];
    }

    /**
     * 获取所有标签的总频次
     *
     * @return
     */
    public int getTotalFrequency()
    {
        return totalFrequency;
    }

    protected E convert(String label)
    {
        return Enum.valueOf(enumType, label);
    }

    /**
     * 拓展内部矩阵,仅用于通过反射新增了枚举实例之后的兼容措施
     */
    public void extendSize()
    {
        ++ordinaryMax;
        double[][] n_transititon_probability = new double[ordinaryMax][ordinaryMax];
        for (int i = 0; i < transititon_probability.length; i++)
        {
            System.arraycopy(transititon_probability[i], 0, n_transititon_probability[i], 0, transititon_probability.length);
        }
        transititon_probability = n_transititon_probability;

        int[] n_total = new int[ordinaryMax];
        System.arraycopy(total, 0, n_total, 0, total.length);
        total = n_total;

        double[] n_start_probability = new double[ordinaryMax];
        System.arraycopy(start_probability, 0, n_start_probability, 0, start_probability.length);
        start_probability = n_start_probability;

        int[][] n_matrix = new int[ordinaryMax][ordinaryMax];
        for (int i = 0; i < matrix.length; i++)
        {
            System.arraycopy(matrix[i], 0, n_matrix[i], 0, matrix.length);
        }
        matrix = n_matrix;
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder("TransformMatrixDictionary{");
        sb.append("enumType=").append(enumType);
        sb.append(", ordinaryMax=").append(ordinaryMax);
        sb.append(", matrix=").append(Arrays.toString(matrix));
        sb.append(", total=").append(Arrays.toString(total));
        sb.append(", totalFrequency=").append(totalFrequency);
        sb.append('}');
        return sb.toString();
    }
}
