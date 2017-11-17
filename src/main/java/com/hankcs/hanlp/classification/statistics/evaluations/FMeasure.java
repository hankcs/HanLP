/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/17 PM3:11</create-date>
 *
 * <copyright file="FMeasure.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.statistics.evaluations;

import java.io.Serializable;

public class FMeasure implements Serializable
{
    /**
     * 测试样本空间
     */
    int size;
    /**
     * 平均准确率
     */
    public double average_accuracy;
    /**
     * 平均精确率
     */
    public double average_precision;
    /**
     * 平均召回率
     */
    public double average_recall;
    /**
     * 平均F1
     */
    public double average_f1;

    /**
     * 分类准确率
     */
    public double accuracy[];
    /**
     * 分类精确率
     */
    public double precision[];
    /**
     * 分类召回率
     */
    public double recall[];
    /**
     * 分类F1
     */
    public double[] f1;
    /**
     * 分类名称
     */
    public String[] catalog;

    /**
     * 速度
     */
    public double speed;

    @Override
    public String toString()
    {
        int l = -1;
        for (String c : catalog)
        {
            l = Math.max(l, c.length());
        }
        final int w = 6;
        final StringBuilder sb = new StringBuilder(10000);

        printf(sb, "%*s\t%*s\t%*s\t%*s\t%*s%n".replace('*', Character.forDigit(w, 10)), "P", "R", "F1", "A", "");
        for (int i = 0; i < catalog.length; i++)
        {
            printf(sb, ("%*.2f\t%*.2f\t%*.2f\t%*.2f\t%"+l+"s%n").replace('*', Character.forDigit(w, 10)),
                   precision[i] * 100.,
                   recall[i] * 100.,
                   f1[i] * 100.,
                   accuracy[i] * 100.,
                   catalog[i]);
        }
        printf(sb, ("%*.2f\t%*.2f\t%*.2f\t%*.2f\t%"+l+"s%n").replace('*', Character.forDigit(w, 10)),
               average_precision * 100.,
               average_recall * 100.,
               average_f1 * 100.,
               average_accuracy * 100.,
               "avg.");
        printf(sb, "data size = %d, speed = %.2f doc/s\n", size, speed);
        return sb.toString();
    }

    private static void printf(StringBuilder sb, String format, Object... args)
    {
        sb.append(String.format(format, args));
    }
}
