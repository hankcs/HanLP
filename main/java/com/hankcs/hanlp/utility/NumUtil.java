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
            if (word.length() == 2 && TextUtility.isInAggregate("第上成±—＋∶·．／", word))
                return true;
            if (word.length() == 1 && TextUtility.isInAggregate("+-./", word))
                return true;
        }
        return false;
    }
}
