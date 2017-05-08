/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/5 17:01</create-date>
 *
 * <copyright file="CharArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest.scorer.editdistance;

import com.hankcs.hanlp.algorithm.EditDistance;
import com.hankcs.hanlp.suggest.scorer.ISentenceKey;

/**
 * 对字符数组的封装，可以代替String
 * @author hankcs
 */
public class CharArray implements Comparable<CharArray>, ISentenceKey<CharArray>
{
    char[] value;

    public CharArray(char[] value)
    {
        this.value = value;
    }

    @Override
    public int compareTo(CharArray other)
    {
        int len1 = value.length;
        int len2 = other.value.length;
        int lim = Math.min(len1, len2);
        char v1[] = value;
        char v2[] = other.value;

        int k = 0;
        while (k < lim)
        {
            char c1 = v1[k];
            char c2 = v2[k];
            if (c1 != c2)
            {
                return c1 - c2;
            }
            k++;
        }
        return len1 - len2;
    }

    @Override
    public Double similarity(CharArray other)
    {
        int distance = EditDistance.compute(this.value, other.value) + 1;
        return 1.0 / distance;
    }
}
