/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/16 AM11:17</create-date>
 *
 * <copyright file="ConsoleLogger.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.utilities.io;

/**
 * 输出到stdout和stderr的日志系统
 *
 * @author hankcs
 */
public class ConsoleLogger implements ILogger
{
    /**
     * 默认日志
     */
    public static ILogger logger = new ConsoleLogger();
    long start;

    public void out(String format, Object... args)
    {
        System.out.printf(format, args);
    }

    public void err(String format, Object... args)
    {
        System.err.printf(format, args);
    }

    public void start(String format, Object... args)
    {
        out(format, args);
        start = System.currentTimeMillis();
    }

    public void finish(String format, Object... args)
    {
        out(String.format("耗时 %d ms", System.currentTimeMillis() - start) + format, args);
    }
}