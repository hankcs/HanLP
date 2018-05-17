
package com.hankcs.hanlp.mining.word2vec;

import java.io.*;

/**
 * 一些工具方法
 */
final class Utility
{
    private static final int SECOND = 1000;
    private static final int MINUTE = 60 * SECOND;
    private static final int HOUR = 60 * MINUTE;
    private static final int DAY = 24 * HOUR;

    static String humanTime(long ms)
    {
        StringBuffer text = new StringBuffer("");
        if (ms > DAY)
        {
            text.append(ms / DAY).append(" d ");
            ms %= DAY;
        }
        if (ms > HOUR)
        {
            text.append(ms / HOUR).append(" h ");
            ms %= HOUR;
        }
        if (ms > MINUTE)
        {
            text.append(ms / MINUTE).append(" m ");
            ms %= MINUTE;
        }
        if (ms > SECOND)
        {
            long s = ms / SECOND;
            if (s < 10)
            {
                text.append('0');
            }
            text.append(s).append(" s ");
//            ms %= SECOND;
        }
//        text.append(ms + " ms");

        return text.toString();
    }

    /**
     * @param c
     */
    public static void closeQuietly(Closeable c)
    {
        try
        {
            if (c != null) c.close();
        }
        catch (IOException ignored)
        {
        }
    }

    /**
     * @param raf
     */
    public static void closeQuietly(RandomAccessFile raf)
    {
        try
        {
            if (raf != null) raf.close();
        }
        catch (IOException ignored)
        {
        }
    }

    public static void closeQuietly(InputStream is)
    {
        try
        {
            if (is != null) is.close();
        }
        catch (IOException ignored)
        {
        }
    }

    public static void closeQuietly(Reader r)
    {
        try
        {
            if (r != null) r.close();
        }
        catch (IOException ignored)
        {
        }
    }

    public static void closeQuietly(OutputStream os)
    {
        try
        {
            if (os != null) os.close();
        }
        catch (IOException ignored)
        {
        }
    }

    public static void closeQuietly(Writer w)
    {
        try
        {
            if (w != null) w.close();
        }
        catch (IOException ignored)
        {
        }
    }

    /**
     * 数组分割
     *
     * @param from 源
     * @param to   目标
     * @param <T>  类型
     * @return 目标
     */
    public static <T> T[] shrink(T[] from, T[] to)
    {
        assert to.length <= from.length;
        System.arraycopy(from, 0, to, 0, to.length);
        return to;
    }
}
