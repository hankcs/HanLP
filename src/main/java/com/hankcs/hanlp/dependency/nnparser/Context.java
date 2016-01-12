/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/1 22:02</create-date>
 *
 * <copyright file="Context.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

/**
 * 上下文
 * @author hankcs
 */
public class Context
{
    int S0, S1, S2, N0, N1, N2;             // 栈和队列的第i个元素
    int S0L, S0R, S0L2, S0R2, S0LL, S0RR;   // 该元素最左右的子节点、倒数第2个最左右的子节点
    int S1L, S1R, S1L2, S1R2, S1LL, S1RR;
}
