package com.hankcs.hanlp.algorithm.ahocorasick.interval;

/**
 * 区间接口
 */
public interface Intervalable extends Comparable
{
    /**
     * 起点
     * @return
     */
    public int getStart();

    /**
     * 终点
     * @return
     */
    public int getEnd();

    /**
     * 长度
     * @return
     */
    public int size();

}
