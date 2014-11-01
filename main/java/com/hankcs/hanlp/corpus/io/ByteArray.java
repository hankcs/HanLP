/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/30 14:33</create-date>
 *
 * <copyright file="ByteArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import com.hankcs.hanlp.utility.Utility;

/**
 * 对字节数组进行封装，提供方便的读取操作
 * @author hankcs
 */
public class ByteArray
{
    byte[] bytes;
    int offset;

    public ByteArray(byte[] bytes)
    {
        this.bytes = bytes;
    }

    /**
     * 从文件读取一个字节数组
     * @param path
     * @return
     */
    public static ByteArray createByteArray(String path)
    {
        byte[] bytes = IOUtil.readBytes(path);
        if (bytes == null) return null;
        return new ByteArray(bytes);
    }


    /**
     * 读取一个int
     * @return
     */
    public int nextInt()
    {
        int result = Utility.bytesHighFirstToInt(bytes, offset);
        offset += 4;
        return result;
    }

    /**
     * 读取一个char
     * @return
     */
    public char nextChar()
    {
        char result = Utility.bytesHighFirstToChar(bytes, offset);
        offset += 2;
        return result;
    }

    public boolean hasMore()
    {
        return offset < bytes.length;
    }
}
