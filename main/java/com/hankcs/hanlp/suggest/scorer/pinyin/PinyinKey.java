/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/5 20:49</create-date>
 *
 * <copyright file="PinyinKey.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.suggest.scorer.pinyin;

import com.hankcs.hanlp.algoritm.EditDistance;
import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.String2PinyinConverter;
import com.hankcs.hanlp.suggest.scorer.ISentenceKey;

/**
 * @author hankcs
 */
public class PinyinKey implements Comparable<PinyinKey>, ISentenceKey<PinyinKey>
{
    /**
     * 句子的拼音
     */
    Pinyin[] pinyinArray;
    /**
     * 拼音的ordinal数组
     */
    int[] pyOrdinalArray;
    /**
     * 输入法头数组
     */
    int[] headOrdinalArray;
    /**
     * 首字母数组
     */
    char[] firstCharArray;

    public PinyinKey(String sentence)
    {
        pinyinArray = String2PinyinConverter.convert2Array(sentence, true);
        pyOrdinalArray = new int[pinyinArray.length];
        headOrdinalArray = new int[pinyinArray.length];
        firstCharArray = new char[pinyinArray.length];
        for (int i = 0; i < pyOrdinalArray.length; ++i)
        {
            pyOrdinalArray[i] = pinyinArray[i].ordinal();
            headOrdinalArray[i] = pinyinArray[i].getHead().ordinal();
            firstCharArray[i] = pinyinArray[i].getFirstChar();
        }
    }

    @Override
    public int compareTo(PinyinKey o)
    {
        int len1 = pyOrdinalArray.length;
        int len2 = o.pyOrdinalArray.length;
        int lim = Math.min(len1, len2);
        int[] v1 = pyOrdinalArray;
        int[] v2 = o.pyOrdinalArray;

        int k = 0;
        while (k < lim)
        {
            int c1 = v1[k];
            int c2 = v2[k];
            if (c1 != c2)
            {
                return c1 - c2;
            }
            k++;
        }
        return len1 - len2;
    }

    @Override
    public Double similarity(PinyinKey other)
    {
        return 3.0 / (EditDistance.compute(pyOrdinalArray, other.pyOrdinalArray) + EditDistance.compute(headOrdinalArray, other.headOrdinalArray) + EditDistance.compute(firstCharArray, other.firstCharArray) + 1);
    }

    /**
     * 拼音的个数
     * @return
     */
    public int size()
    {
        int length = 0;
        for (Pinyin pinyin : pinyinArray)
        {
            if (pinyin != Pinyin.none5) ++length;
        }

        return length;
    }
}
