/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/2/26 0:59</create-date>
 *
 * <copyright file="NumberAutomata.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.algoritm.automata.fo;

import com.hankcs.hanlp.algoritm.automata.IBiAutomata;

/**
 * 可以识别数词成分的“自动机”
 * @author hankcs
 */
public class NumberAutomata implements IBiAutomata
{
    static byte[] isNum = new byte[Character.MAX_VALUE];
    static
    {
        String number = "第特新自编附首甲乙丙丁学侧负门边上01234567890０１２３４５６７８９零一二三四五六七八九十ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
                "ＡＢＣＤＥＦＧ—－-~#、东南西北";
        for (int i = 0; i < number.length(); ++i)
        {
            isNum[number.charAt(i)] = 1;
        }
    }

    @Override
    public boolean transmit(int from, int to)
    {
        return isNum[from] > 0 && isNum[to] > 0;
    }
}
