/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-09-18 下午7:51</create-date>
 *
 * <copyright file="PosTag.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.utility;

import com.hankcs.hanlp.corpus.tag.Nature;

import java.util.*;

/**
 * @author hankcs
 */
public class PosTagUtility
{
    public static Set<String> tagSet;

    static
    {
        tagSet = new LinkedHashSet<String>();
        String allTags = "a " +
                "b " +
                "c " +
                "d " +
                "e " +
                "f " +
                "h " +
                "i " +
                "j " +
                "k " +
                "l " +
                "m " +
                "n " +
                "nr " +
                "ns " +
                "nt " +
                "nu " +
                "ne " +
                "nz " +
                "o " +
                "p " +
                "q " +
                "r " +
                "s " +
                "t " +
                "u " +
                "v " +
                "w " +
                "x " +
                "y ";

        String[] tagArray = allTags.split("\\s+");
        Arrays.sort(tagArray, new Comparator<String>()
        {
            @Override
            public int compare(String o1, String o2)
            {
                return new Integer(o2.length()).compareTo(o1.length());
            }
        });
        for (String tag : tagArray)
        {
            tagSet.add(tag);
        }
    }

    public static String convert(String tag)
    {
        for (String t : tagSet)
        {
            if (tag.startsWith(t))
            {
                return t;
            }
        }

        return "n";
    }

    public static String convert(Nature tag)
    {
        return convert(tag.toString());
    }
}
