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

import com.hankcs.hanlp.utility.ByteUtil;
import com.hankcs.hanlp.utility.TextUtility;

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
        int result = TextUtility.bytesHighFirstToInt(bytes, offset);
        offset += 4;
        return result;
    }

    public double nextDouble()
    {
        double result = ByteUtil.bytesHighFirstToDouble(bytes, offset);
        offset += 8;
        return result;
    }

    /**
     * 读取一个char，对应于writeChar
     * @return
     */
    public char nextChar()
    {
        char result = TextUtility.bytesHighFirstToChar(bytes, offset);
        offset += 2;
        return result;
    }

    /**
     * 读取一个字节
     * @return
     */
    public byte nextByte()
    {
        return bytes[offset++];
    }

    public boolean hasMore()
    {
        return offset < bytes.length;
    }

    /**
     * 读取一个String，注意这个String是双字节版的，在字符之前有一个整型表示长度
     * @return
     */
    public String nextString()
    {
        StringBuilder sb = new StringBuilder();
        int length = nextInt();
        for (int i = 0; i < length; ++i)
        {
            sb.append(nextChar());
        }
        return sb.toString();
    }

    public float nextFloat()
    {
        float result = TextUtility.bytesHighFirstToFloat(bytes, offset);
        offset += 4;
        return result;
    }
}
