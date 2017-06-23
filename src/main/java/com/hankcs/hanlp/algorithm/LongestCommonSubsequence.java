/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/7 11:29</create-date>
 *
 * <copyright file="LongestCommonSubsequence.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algorithm;

/**
 * 最长公共子序列（Longest Common Subsequence）指的是两个字符串中的最长公共子序列，不要求子序列连续。
 * @author hankcs
 */
public class LongestCommonSubsequence
{
    public static int compute(char[] str1, char[] str2)
    {
        int substringLength1 = str1.length;
        int substringLength2 = str2.length;

        // 构造二维数组记录子问题A[i]和B[j]的LCS的长度
        int[][] opt = new int[substringLength1 + 1][substringLength2 + 1];

        // 从后向前，动态规划计算所有子问题。也可从前到后。
        for (int i = substringLength1 - 1; i >= 0; i--)
        {
            for (int j = substringLength2 - 1; j >= 0; j--)
            {
                if (str1[i] == str2[j])
                    opt[i][j] = opt[i + 1][j + 1] + 1;// 状态转移方程
                else
                    opt[i][j] = Math.max(opt[i + 1][j], opt[i][j + 1]);// 状态转移方程
            }
        }
//        System.out.println("substring1:" + new String(str1));
//        System.out.println("substring2:" + new String(str2));
//        System.out.print("LCS:");

//        int i = 0, j = 0;
//        while (i < substringLength1 && j < substringLength2)
//        {
//            if (str1[i] == str2[j])
//            {
//                System.out.print(str1[i]);
//                i++;
//                j++;
//            }
//            else if (opt[i + 1][j] >= opt[i][j + 1])
//                i++;
//            else
//                j++;
//        }
//        System.out.println();
        return opt[0][0];
    }

    public static int compute(String str1, String str2)
    {
        return compute(str1.toCharArray(), str2.toCharArray());
    }
}
