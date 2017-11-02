package com.hankcs.hanlp.classification.tokenizers;

/**
 * @author hankcs
 */
public class CharType
{

    /**
     * 中文字符
     */
    public static final byte CT_CHINESE = 1;

    /**
     * 字母
     */
    public static final byte CT_LETTER = 2;

    /**
     * 数字
     */
    public static final byte CT_NUM = 3;


    static byte[] type;

    static
    {
        type = new byte[65536];
        for (int i = 19968; i < 40870; ++i)
        {
            type[i] = CT_CHINESE;
        }
        for (char c : "0123456789".toCharArray())
        {
            type[c] = CT_NUM;
        }
        for (char c : "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray())
        {
            type[c] = CT_LETTER;
        }
    }

    /**
     * 获取字符的类型
     * @param c
     * @return
     */
    public static byte get(char c)
    {
        return type[(int)c];
    }
}
