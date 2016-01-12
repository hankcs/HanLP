/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/7/29 16:37</create-date>
 *
 * <copyright file="DumpHander.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public abstract class LineHandler
{
    String delimiter = "\t";

    public LineHandler(String delimiter)
    {
        this.delimiter = delimiter;
    }

    public LineHandler()
    {
    }

    public void handle(String line) throws Exception
    {
        List<String> tokenList = new LinkedList<String>();
        int start = 0;
        int end;
        while ((end = line.indexOf(delimiter, start)) != -1)
        {
            tokenList.add(line.substring(start, end));
            start = end + 1;
        }
        tokenList.add(line.substring(start, line.length()));
        handle(tokenList.toArray(new String[0]));
    }

    public void done() throws IOException
    {
        // do noting
    }

    public abstract void handle(String[] params) throws IOException;
}
