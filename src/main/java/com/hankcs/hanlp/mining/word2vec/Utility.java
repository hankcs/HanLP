
package com.hankcs.hanlp.mining.word2vec;

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
}
