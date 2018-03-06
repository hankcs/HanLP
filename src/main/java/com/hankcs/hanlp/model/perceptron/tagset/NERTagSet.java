/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-28 11:40</create-date>
 *
 * <copyright file="NERTagSet.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.tagset;

import com.hankcs.hanlp.model.perceptron.common.TaskType;

/**
 * @author hankcs
 */
public class NERTagSet extends TagSet
{
    public final String O_TAG = "O";
    public final char O_TAG_CHAR = 'O';
    public final String B_TAG_PREFIX = "B-";
    public final char B_TAG_CHAR = 'B';
    public final String M_TAG_PREFIX = "M-";
    public final String E_TAG_PREFIX = "E-";
    public final String S_TAG = "S";
    public final char S_TAG_CHAR = 'S';

    /**
     * 非NER
     */
    public final int O;

    public NERTagSet()
    {
        super(TaskType.NER);
        O = add(O_TAG);
    }

    public static String posOf(String tag)
    {
        int index = tag.indexOf('-');
        if (index == -1)
        {
            return tag;
        }

        return tag.substring(index + 1);
    }
}
