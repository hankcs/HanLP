/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/21 20:13</create-date>
 *
 * <copyright file="PathNode.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

/**
 * 路径上的节点
 * @author hankcs
 */
public class PathNode
{
    /**
     * 节点前驱
     */
    public int from;
    /**
     * 节点在顶点数组中的下标
     */
    public int index;

    /**
     * 构造一个节点
     * @param from 节点前驱
     * @param index 节点在顶点数组中的下标
     */
    public PathNode(int from, int index)
    {
        this.from = from;
        this.index = index;
    }

    @Override
    public String toString()
    {
        return "PathNode{" +
                "from=" + from +
                ", index=" + index +
                '}';
    }
}
