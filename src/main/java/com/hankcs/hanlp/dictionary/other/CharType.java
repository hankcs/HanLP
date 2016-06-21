/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/5 15:37</create-date>
 *
 * <copyright file="CharType.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.other;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.ByteArray;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 字符类型
 * @author hankcs
 */
public class CharType
{
    /**
     * 单字节
     */
    public static final byte CT_SINGLE = 5;

    /**
     * 分隔符"!,.?()[]{}+=
     */
    public static final byte CT_DELIMITER = CT_SINGLE + 1;

    /**
     * 中文字符
     */
    public static final byte CT_CHINESE = CT_SINGLE + 2;

    /**
     * 字母
     */
    public static final byte CT_LETTER = CT_SINGLE + 3;

    /**
     * 数字
     */
    public static final byte CT_NUM = CT_SINGLE + 4;

    /**
     * 序号
     */
    public static final byte CT_INDEX = CT_SINGLE + 5;

    /**
     * 其他
     */
    public static final byte CT_OTHER = CT_SINGLE + 12;
    
    public static byte[] type;

    static
    {
        type = new byte[65536];
        logger.info("字符类型对应表开始加载 " + HanLP.Config.CharTypePath);
        long start = System.currentTimeMillis();
        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.CharTypePath);
        if (byteArray == null)
        {
            System.err.println("字符类型对应表加载失败：" + HanLP.Config.CharTypePath);
            System.exit(-1);
        }
        else
        {
            while (byteArray.hasMore())
            {
                int b = byteArray.nextChar();
                int e = byteArray.nextChar();
                byte t = byteArray.nextByte();
                for (int i = b; i <= e; ++i)
                {
                    type[i] = t;
                }
            }
            logger.info("字符类型对应表加载成功，耗时" + (System.currentTimeMillis() - start) + " ms");
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
