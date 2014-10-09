package com.hankcs.hanlp.utility;

/**
 * 数字相关的工具类
 *
 * @author sinboy
 * @since 2007.5.23
 */
public class NumUtil
{
    /**
     * 字符串的所有单个字符都是数字但本身并不是一个有意义的数字
     *
     * @param word
     * @return
     */
    public static boolean isNumStrNotNum(String word)
    {
        if (word != null)
        {
            if (word.length() == 2 && Utility.isInAggregate("第上成±—＋∶·．／", word))
                return true;
            if (word.length() == 1 && Utility.isInAggregate("+-./", word))
                return true;
        }
        return false;
    }

    /**
     * 是否是数字、连字符的情况，如：３-4月
     *
     * @param pos
     * @param str
     * @return
     */
    public static boolean isNumDelimiter(int pos, String str)
    {
        if (str != null && str.length() > 1)
        {
            String first = str.substring(0, 1);
            // //27904='m'*256 29696='t'*256
            if ((Math.abs(pos) == POSTag.NUM || Math.abs(pos) == POSTag.TIME)
                    && ("—".equals(first) || "-".equals(first)))
                return true;
        }
        return false;
    }

}
