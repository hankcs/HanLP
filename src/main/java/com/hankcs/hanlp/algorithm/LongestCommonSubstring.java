/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/7 10:10</create-date>
 *
 * <copyright file="ArrayCount.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algorithm;

/**
 * 求最长公共字串的长度<br>
 *     最长公共子串（Longest Common Substring）指的是两个字符串中的最长公共子串，要求子串一定连续
 *
 * @author hankcs
 */
public class LongestCommonSubstring
{
    public static int compute(char[] str1, char[] str2)
    {
        int size1 = str1.length;
        int size2 = str2.length;
        if (size1 == 0 || size2 == 0) return 0;

        // the start position of substring in original string
//        int start1 = -1;
//        int start2 = -1;
        // the longest length of com.hankcs.common substring
        int longest = 0;

        // record how many comparisons the solution did;
        // it can be used to know which algorithm is better
//        int comparisons = 0;

        for (int i = 0; i < size1; ++i)
        {
            int m = i;
            int n = 0;
            int length = 0;
            while (m < size1 && n < size2)
            {
//                ++comparisons;
                if (str1[m] != str2[n])
                {
                    length = 0;
                }
                else
                {
                    ++length;
                    if (longest < length)
                    {
                        longest = length;
//                        start1 = m - longest + 1;
//                        start2 = n - longest + 1;
                    }
                }

                ++m;
                ++n;
            }
        }

        // shift string2 to find the longest com.hankcs.common substring
        for (int j = 1; j < size2; ++j)
        {
            int m = 0;
            int n = j;
            int length = 0;
            while (m < size1 && n < size2)
            {
//                ++comparisons;
                if (str1[m] != str2[n])
                {
                    length = 0;
                }
                else
                {
                    ++length;
                    if (longest < length)
                    {
                        longest = length;
//                        start1 = m - longest + 1;
//                        start2 = n - longest + 1;
                    }
                }

                ++m;
                ++n;
            }
        }
//        System.out.printf("from %d of %s and %d of %s, compared for %d times\n", start1, new String(str1), start2, new String(str2), comparisons);
        return longest;
    }

    public static int compute(String str1, String str2)
    {
        return compute(str1.toCharArray(), str2.toCharArray());
    }
}
