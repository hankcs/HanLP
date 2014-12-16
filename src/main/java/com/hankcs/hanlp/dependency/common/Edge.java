/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 17:40</create-date>
 *
 * <copyright file="Edge.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.common;

/**
 * 一条边
 *
 * @author hankcs
 */
public class Edge
{
    public int from;
    public int to;
    public float cost;
    public String label;

    public Edge(int from, int to, String label, float cost)
    {
        this.from = from;
        this.to = to;
        this.cost = cost;
        this.label = label;
    }
}
