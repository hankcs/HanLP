/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-03 上午10:23</create-date>
 *
 * <copyright file="CWSEvaluator.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.common;

import com.hankcs.hanlp.corpus.io.IOUtil;

import java.io.IOException;
import java.util.*;

/**
 * 中文分词评测工具
 *
 * @author hankcs
 */
public class CWSEvaluator
{
    private int A_size, B_size, A_cap_B_size, OOV, OOV_R, IV, IV_R;
    private Set<String> dic;

    public CWSEvaluator()
    {
    }

    public CWSEvaluator(Set<String> dic)
    {
        this.dic = dic;
    }

    public CWSEvaluator(String dictPath) throws IOException
    {
        this(new TreeSet<String>());
        if (dictPath == null) return;
        try
        {
            IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(dictPath);
            for (String word : lineIterator)
            {
                word = word.trim();
                if (word.isEmpty()) continue;
                dic.add(word);
            }
        }
        catch (Exception e)
        {
            throw new IOException(e);
        }
    }

    /**
     * 获取PRF
     *
     * @param percentage 百分制
     * @return
     */
    public Result getResult(boolean percentage)
    {
        float p = A_cap_B_size / (float) B_size;
        float r = A_cap_B_size / (float) A_size;
        if (percentage)
        {
            p *= 100;
            r *= 100;
        }
        float oov_r = Float.NaN;
        if (OOV > 0)
        {
            oov_r = OOV_R / (float) OOV;
            if (percentage)
                oov_r *= 100;
        }
        float iv_r = Float.NaN;
        if (IV > 0)
        {
            iv_r = IV_R / (float) IV;
            if (percentage)
                iv_r *= 100;
        }
        return new Result(p, r, 2 * p * r / (p + r), oov_r, iv_r);
    }


    /**
     * 获取PRF
     *
     * @return
     */
    public Result getResult()
    {
        return getResult(true);
    }

    /**
     * 将分词结果转换为区间
     *
     * @param line 商品 和 服务
     * @return [(0, 2), (2, 3), (3, 5)]
     */
    public static List<Region> toRegion(String line)
    {
        List<Region> region = new LinkedList<Region>();
        int start = 0;
        for (String word : line.split("\\s+"))
        {
            int end = start + word.length();
            region.add(new Region(start, end));
            start = end;
        }
        return region;
    }

    /**
     * 比较标准答案与分词结果
     *
     * @param gold
     * @param pred
     */
    public void compare(String gold, String pred)
    {

        Set<Region> A = new TreeSet<Region>(toRegion(gold));
        Set<Region> B = new TreeSet<Region>(toRegion(pred));
        A_size += A.size();
        B_size += B.size();
        B.retainAll(A);
        A_cap_B_size += B.size();
        if (dic != null)
        {
            String text = gold.replaceAll("\\s+", "");
            for (Region region : A)
            {
                if (dic.contains(region.toString(text)))
                    IV += 1;
                else
                    OOV += 1;
            }
            for (Region region : B)
            {
                if (dic.contains(region.toString(text)))
                    IV_R += 1;
                else
                    OOV_R += 1;
            }
        }
    }

    /**
     * 在标准答案与分词结果上执行评测
     *
     * @param goldFile
     * @param predFile
     * @return
     */
    public static Result evaluate(String goldFile, String predFile) throws IOException
    {
        return evaluate(goldFile, predFile, null);
    }

    /**
     * 在标准答案与分词结果上执行评测
     *
     * @param goldFile
     * @param predFile
     * @return
     */
    public static Result evaluate(String goldFile, String predFile, String dictPath) throws IOException
    {
        IOUtil.LineIterator goldIter = new IOUtil.LineIterator(goldFile);
        IOUtil.LineIterator predIter = new IOUtil.LineIterator(predFile);
        CWSEvaluator evaluator = new CWSEvaluator(dictPath);
        while (goldIter.hasNext() && predIter.hasNext())
        {
            evaluator.compare(goldIter.next(), predIter.next());
        }
        return evaluator.getResult();
    }

    private static class Region implements Comparable<Region>
    {
        int start, end;

        public Region(int start, int end)
        {
            this.start = start;
            this.end = end;
        }

        @Override
        public int compareTo(Region o)
        {
            if (start != o.start)
                return new Integer(start).compareTo(o.start);
            return new Integer(end).compareTo(o.end);
        }

        public String toString(String text)
        {
            return text.substring(start, end);
        }
    }

    public static class Result
    {
        float P, R, F1, OOV_R, IV_R;

        public Result(float p, float r, float f1, float OOV_R, float IV_R)
        {
            P = p;
            R = r;
            F1 = f1;
            this.OOV_R = OOV_R;
            this.IV_R = IV_R;
        }

        @Override
        public String toString()
        {
            final StringBuilder sb = new StringBuilder();
            sb.append("P=").append(P);
            sb.append(", R=").append(R);
            sb.append(", F1=").append(F1);
            sb.append(", OOV_R=").append(OOV_R);
            sb.append(", IV_R=").append(IV_R);
            return sb.toString();
        }
    }
}
