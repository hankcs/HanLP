package com.hankcs.hanlp.algorithm.ahocorasick.interval;

import java.util.Comparator;

/**
 * 按照长度比较区间，如果长度相同，则比较起点
 */
public class IntervalableComparatorBySize implements Comparator<Intervalable>
{
    @Override
    public int compare(Intervalable intervalable, Intervalable intervalable2)
    {
        int comparison = intervalable2.size() - intervalable.size();
        if (comparison == 0)
        {
            comparison = intervalable.getStart() - intervalable2.getStart();
        }
        return comparison;
    }
}
