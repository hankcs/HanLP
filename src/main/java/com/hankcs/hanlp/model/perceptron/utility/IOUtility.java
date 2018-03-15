/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM7:29</create-date>
 *
 * <copyright file="IOUtility.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.utility;

import com.hankcs.hanlp.corpus.io.IOUtil;

import java.util.regex.Pattern;

/**
 * @author hankcs
 */
public class IOUtility extends IOUtil
{
    private static Pattern PATTERN_SPACE = Pattern.compile("\\s+");
    public static String[] readLineToArray(String line)
    {
        line = line.trim();
        if (line.length() == 0) return new String[0];
        return PATTERN_SPACE.split(line);
    }
}
