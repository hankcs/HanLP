package com.hankcs.hanlp.algorithm.ahocorasick.interval;

import java.util.Comparator;

/**
 * 按起点比较区间
 */
public class IntervalableComparatorByPosition implements Comparator<Intervalable>
{
    @Override
    public int compare(Intervalable intervalable, Intervalable intervalable2)
    {
        return intervalable.getStart() - intervalable2.getStart();
    }
}
