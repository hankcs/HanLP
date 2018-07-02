/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM5:28</create-date>
 *
 * <copyright file="Tag.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.tagset;

import com.hankcs.hanlp.model.perceptron.common.TaskType;

/**
 * @author hankcs
 */
public class CWSTagSet extends TagSet
{
    public final int B;
    public final int M;
    public final int E;
    public final int S;

    public CWSTagSet(int b, int m, int e, int s)
    {
        super(TaskType.CWS);
        B = b;
        M = m;
        E = e;
        S = s;
        String[] id2tag = new String[4];
        id2tag[b] = "B";
        id2tag[m] = "M";
        id2tag[e] = "E";
        id2tag[s] = "S";
        for (String tag : id2tag)
        {
            add(tag);
        }
        lock();
    }

    public CWSTagSet()
    {
        super(TaskType.CWS);
        B = add("B");
        M = add("M");
        E = add("E");
        S = add("S");
        lock();
    }
}
