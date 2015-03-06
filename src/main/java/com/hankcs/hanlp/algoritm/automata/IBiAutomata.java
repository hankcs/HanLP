/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/2/17 22:16</create-date>
 *
 * <copyright file="IBiAutomata.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algoritm.automata;

/**
 * 二元自动机接口
 * @author hankcs
 */
public interface IBiAutomata
{
    /**
     * 转移状态
     * @param from 当前状态
     * @param to 目标状态
     * @return 如果可以转移则返回to，否则返回负数
     */
    boolean transmit(int from, int to);
}
