/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 20:37</create-date>
 *
 * <copyright file="Action.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.action;

/**
 * @author hankcs
 */
public class Action extends AbstractInexactAction implements ActionType
{
    public Action()
    {
    }

    /**
     * 创建动作
     * @param name 动作名称
     * @param rel 依存关系
     */
    public Action(int name, int rel)
    {
        super(name, rel);
    }
}
