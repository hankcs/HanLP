/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/6 21:32</create-date>
 *
 * <copyright file="ByteArrayStream.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.corpus.io;

import com.hankcs.hanlp.utility.TextUtility;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 流式的字节数组，降低读取时的内存峰值
 * @author hankcs
 */
public class ByteArrayStream extends ByteArray
{
    /**
     * 每次读取1mb
     */
    private int bufferSize;
    private FileChannel fileChannel;

    public ByteArrayStream(byte[] bytes, int bufferSize, FileChannel fileChannel)
    {
        super(bytes);
        this.bufferSize = bufferSize;
        this.fileChannel = fileChannel;
    }

    public static ByteArrayStream createByteArrayStream(String path)
    {
        try
        {
            FileInputStream fileInputStream = new FileInputStream(path);
            FileChannel channel = fileInputStream.getChannel();
            long size = channel.size();
            int bufferSize = (int) Math.min(1048576, size);
            ByteBuffer byteBuffer = ByteBuffer.allocate(bufferSize);
            if (channel.read(byteBuffer) == size)
            {
                channel.close();
                channel = null;
            }
            byteBuffer.flip();
            byte[] bytes = byteBuffer.array();
            return new ByteArrayStream(bytes, bufferSize, channel);
        }
        catch (Exception e)
        {
            logger.warning(TextUtility.exceptionToString(e));
            return null;
        }
    }

    @Override
    public int nextInt()
    {
        ensureAvailableBytes(4);
        return super.nextInt();
    }

    @Override
    public char nextChar()
    {
        ensureAvailableBytes(2);
        return super.nextChar();
    }

    @Override
    public double nextDouble()
    {
        ensureAvailableBytes(8);
        return super.nextDouble();
    }

    @Override
    public byte nextByte()
    {
        ensureAvailableBytes(1);
        return super.nextByte();
    }

    @Override
    public float nextFloat()
    {
        ensureAvailableBytes(4);
        return super.nextFloat();
    }

    @Override
    public boolean hasMore()
    {
        return offset < bufferSize || fileChannel != null;
    }

    /**
     * 确保buffer数组余有size个字节
     * @param size
     */
    private void ensureAvailableBytes(int size)
    {
        if (offset + size > bufferSize)
        {
            try
            {
                int availableBytes = (int) (fileChannel.size() - fileChannel.position());
                ByteBuffer byteBuffer = ByteBuffer.allocate(Math.min(availableBytes, offset));
                int readBytes = fileChannel.read(byteBuffer);
                if (readBytes == availableBytes)
                {
                    fileChannel.close();
                    fileChannel = null;
                }
                assert readBytes > 0 : "已到达文件尾部！";
                byteBuffer.flip();
                byte[] bytes = byteBuffer.array();
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
}
