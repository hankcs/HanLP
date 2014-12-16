/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 18:28</create-date>
 *
 * <copyright file="State.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.common;

/**
 * @author hankcs
 */
public class State implements Comparable<State>
{
    public float cost;
    public int id;
    public Edge edge;

    public State(float cost, int id, Edge edge)
    {
        this.cost = cost;
        this.id = id;
        this.edge = edge;
    }

    @Override
    public int compareTo(State o)
    {
        return Float.compare(cost, o.cost);
    }
}
