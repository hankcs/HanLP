/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/21 19:33</create-date>
 *
 * <copyright file="Edge.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.common;

/**
 * 基础边，不允许构造
 * @author hankcs
 */
public class Edge
{
    /**
     * 花费
     */
    public double weight;
    /**
     * 节点名字，调试用
     */
    String name;

    protected Edge(double weight, String name)
    {
        this.weight = weight;
        this.name = name;
    }
}
