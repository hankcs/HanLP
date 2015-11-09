/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/31 20:38</create-date>
 *
 * <copyright file="ActionType.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.action;

/**
 * arc-standard system (Nivre, 2004) 用到的动作，类似于 Yamada 和 Matsumoto 提出的分析动作
 * @author hankcs
 */
public interface ActionType
{
    /**
     * 无效动作，正常情况下不会用到
     */
    int kNone = 0;  //! Placeholder for illegal action.
    /**
     * 不建立依存关系，只转移句法分析的焦点，即新的左焦点词是原来的右焦点词，依此类推。
     */
    int kShift = 1;     //! The index of shift action.
    /**
     * 建立右焦点词依存于左焦点词的依存关系
     */
    int kLeftArc = 2;   //! The index of arc left action.
    /**
     * 建立左焦点词依存于右焦点词的依存关系
     */
    int kRightArc = 3;   //! The index of arc right action.
}
