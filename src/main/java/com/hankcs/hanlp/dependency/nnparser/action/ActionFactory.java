/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 20:42</create-date>
 *
 * <copyright file="ActionFactory.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.action;

/**
 * @author hankcs
 */
public class ActionFactory implements ActionType
{
    /**
     * 不建立依存关系，只转移句法分析的焦点，即新的左焦点词是原来的右焦点词，依此类推。
     * @return
     */
    public static Action make_shift()
    {
        return new Action(kShift, 0);
    }

    /**
     * 建立右焦点词依存于左焦点词的依存关系
     * @param rel 依存关系
     * @return
     */
    public static Action make_left_arc(final int rel)
    {
        return new Action(kLeftArc, rel);
    }

    /**
     * 建立左焦点词依存于右焦点词的依存关系
     * @param rel 依存关系
     * @return
     */
    public static Action make_right_arc(final int rel)
    {
        return new Action(kRightArc, rel);
    }
}
