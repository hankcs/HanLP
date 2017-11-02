package com.hankcs.hanlp.mining.word;

import java.util.Map;
import java.util.TreeMap;

/**
 * 提取出来的词语
 * @author hankcs
 */
public class WordInfo
{
    /**
     * 左邻接字集合
     */
    Map<Character, int[]> left;
    /**
     * 右邻接字集合
     */
    Map<Character, int[]> right;
    /**
     * 词语
     */
    public String text;
    /**
     * 词频
     */
    public int frequency;
    float p;
    float leftEntropy;
    float rightEntropy;
    /**
     * 互信息
     */
    public float aggregation;
    /**
     * 信息熵
     */
    public float entropy;

    WordInfo(String text)
    {
        this.text = text;
        left = new TreeMap<Character, int[]>();
        right = new TreeMap<Character, int[]>();
        aggregation = Float.MAX_VALUE;
    }

    private static void increaseFrequency(char c, Map<Character, int[]> storage)
    {
        int[] freq = storage.get(c);
        if (freq == null)
        {
            freq = new int[]{1};
            storage.put(c, freq);
        }
        else
        {
            ++freq[0];
        }
    }

    private float computeEntropy(Map<Character, int[]> storage)
    {
        float sum = 0;
        for (Map.Entry<Character, int[]> entry : storage.entrySet())
        {
            float p = entry.getValue()[0] / (float) frequency;
            sum -= p * Math.log(p);
        }
        return sum;
    }

    void update(char left, char right)
    {
        ++frequency;
        increaseFrequency(left, this.left);
        increaseFrequency(right, this.right);
    }

    void computeProbabilityEntropy(int length)
    {
        p = frequency / (float) length;
        leftEntropy = computeEntropy(left);
        rightEntropy = computeEntropy(right);
        entropy = Math.min(leftEntropy, rightEntropy);
    }

    void computeAggregation(Map<String, WordInfo> word_cands)
    {
        if (text.length() == 1)
        {
            aggregation = (float) Math.sqrt(p);
            return;
        }
        for (int i = 1; i < text.length(); ++i)
        {
            aggregation = Math.min(aggregation,
                                   p / word_cands.get(text.substring(0, i)).p / word_cands.get(text.substring(i)).p);
        }
    }

    @Override
    public String toString()
    {
        return text;
    }
}
