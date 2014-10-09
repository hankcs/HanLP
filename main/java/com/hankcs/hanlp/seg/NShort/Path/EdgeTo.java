/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/21 18:06</create-date>
 *
 * <copyright file="Edge.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort.Path;

/**
 * 记录了终点的边
 * @author hankcs
 */
public class EdgeTo extends Edge
{
    /**
     * 终点
     */
    int to;

    public EdgeTo(int to, double weight, String name)
    {
        super(weight, name);
        this.to = to;
    }

    @Override
    public String toString()
    {
        return "Edge{" +
                "to=" + to +
                ", weight=" + weight +
                ", name='" + name + '\'' +
                '}';
    }
}
