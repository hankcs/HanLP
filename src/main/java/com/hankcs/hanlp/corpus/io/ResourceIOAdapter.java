/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-24 AM8:34</create-date>
 *
 * <copyright file="ResourceIOAdapter.java" company="码农场">
 * Copyright (c) 2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * 从jar包资源读取文件的适配器
 * @author hankcs
 */
public class ResourceIOAdapter implements IIOAdapter
{
    @Override
    public InputStream open(String path) throws IOException
    {
        return IOUtil.getInputStream(path);
    }

    @Override
    public OutputStream create(String path) throws IOException
    {
        if (IOUtil.isResource(path)) throw new IllegalArgumentException("不支持写入jar包资源路径" + path);
        return new FileOutputStream(path);
    }
}
