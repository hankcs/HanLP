/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-07 PM5:29</create-date>
 *
 * <copyright file="ByteArrayOtherStream.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import com.hankcs.hanlp.utility.TextUtility;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import static com.hankcs.hanlp.HanLP.Config.IOAdapter;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class ByteArrayOtherStream extends ByteArrayStream
{
    InputStream is;

    public ByteArrayOtherStream(byte[] bytes, int bufferSize)
    {
        super(bytes, bufferSize);
    }

    public ByteArrayOtherStream(byte[] bytes, int bufferSize, InputStream is)
    {
        super(bytes, bufferSize);
        this.is = is;
    }

    public static ByteArrayOtherStream createByteArrayOtherStream(String path)
    {
        try
        {
            InputStream is = IOAdapter == null ? new FileInputStream(path) : IOAdapter.open(path);
            return createByteArrayOtherStream(is);
        }
        catch (Exception e)
        {
            logger.warning(TextUtility.exceptionToString(e));
            return null;
        }
    }

    public static ByteArrayOtherStream createByteArrayOtherStream(InputStream is) throws IOException
    {
        int size = is.available();
        int bufferSize = Math.min(1048576, size);
        byte[] bytes = new byte[bufferSize];
        IOUtil.readBytesFromOtherInputStream(is, bytes);
        return new ByteArrayOtherStream(bytes, bufferSize, is);
    }

    @Override
    protected void ensureAvailableBytes(int size)
    {
        if (offset + size > bufferSize)
        {
            try
            {
                int availableBytes = is.available();
                int readBytes = Math.min(availableBytes, offset);
                byte[] bytes = new byte[readBytes];
                IOUtil.readBytesFromOtherInputStream(is, bytes);
                if (readBytes == availableBytes)
                {
                    is.close();
                    is = null;
                }
                assert readBytes > 0 : "已到达文件尾部！";
                System.arraycopy(this.bytes, offset, this.bytes, offset - readBytes, bufferSize - offset);
                System.arraycopy(bytes, 0, this.bytes, bufferSize - readBytes, readBytes);
                offset -= readBytes;
            }
            catch (IOException e)
            {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void close()
    {
        super.close();
        try
        {
            is.close();
        }
        catch (IOException e)
        {
            logger.warning(TextUtility.exceptionToString(e));
        }
    }
}
