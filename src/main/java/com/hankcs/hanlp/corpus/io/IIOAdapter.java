/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-07 PM4:40</create-date>
 *
 * <copyright file="IIOAdapter.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * @author hankcs
 */
public interface IIOAdapter
{
    InputStream open(String path) throws IOException;

    OutputStream save(String path) throws IOException;
}
