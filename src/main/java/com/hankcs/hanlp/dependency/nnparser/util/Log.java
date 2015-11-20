/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/1 20:22</create-date>
 *
 * <copyright file="Log.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser.util;

/**
 * @author hankcs
 */
public class Log
{
    public static void ERROR_LOG(String format, Object ... args)
    {
        System.err.printf(format, args);
    }

    public static void INFO_LOG(String format, Object ... args)
    {
        System.err.printf(format, args);
    }
}
