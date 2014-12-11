/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/26 16:21</create-date>
 *
 * <copyright file="Evaluater.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dependency.CoNll;

import java.text.NumberFormat;

/**
 * 测试工具
 * @author hankcs
 */
public class Evaluator
{
    float U, L, D, A;
    int sentenceCount;
    long start;

    public Evaluator()
    {
        start = System.currentTimeMillis();
    }

    public void e(CoNLLSentence right, CoNLLSentence test)
    {
        ++sentenceCount;
        A += right.word.length;
        for (int i = 0; i < test.word.length; ++i)
        {
            if (test.word[i].HEAD.ID == right.word[i].HEAD.ID)
            {
                ++U;
                if (right.word[i].DEPREL.equals(test.word[i].DEPREL))
                {
                    ++L;
                    if (test.word[i].HEAD.ID != 0)
                    {
                        ++D;
                    }
                }
            }
        }
    }

    public float getUA()
    {
        return U /  A;
    }

    public float getLA()
    {
        return L / A;
    }

    public float getDA()
    {
        return D / (A - sentenceCount);
    }

    @Override
    public String toString()
    {
        NumberFormat percentFormat = NumberFormat.getPercentInstance();
        percentFormat.setMinimumFractionDigits(2);
        StringBuilder sb = new StringBuilder();
        sb.append("UA: ");
        sb.append(percentFormat.format(getUA()));
        sb.append('\t');
        sb.append("LA: ");
        sb.append(percentFormat.format(getLA()));
        sb.append('\t');
        sb.append("DA: ");
        sb.append(percentFormat.format(getDA()));
        sb.append('\t');
        sb.append("sentences: ");
        sb.append(sentenceCount);
        sb.append('\t');
        sb.append("speed: ");
        sb.append(sentenceCount / (float)(System.currentTimeMillis() - start) * 1000);
        sb.append(" sent/s");
        return sb.toString();
    }
}
