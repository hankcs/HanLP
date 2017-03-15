/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/11 15:02</create-date>
 *
 * <copyright file="StringUtils.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.util;

import java.util.regex.Pattern;

/**
 * @author hankcs
 */
public class StringUtils
{

    /**
     * 匹配&或全角状态字符或标点
     */
    public static final String PATTERN = "&|[\uFE30-\uFFA0]|‘’|“”";

    public static String replaceSpecialtyStr(String str, String pattern, String replace)
    {
        if (isBlankOrNull(pattern))
            pattern = "\\s*|\t|\r|\n";//去除字符串中空格、换行、制表
        if (isBlankOrNull(replace))
            replace = "";
        return Pattern.compile(pattern).matcher(str).replaceAll(replace);

    }

    public static boolean isBlankOrNull(String str)
    {
        if (null == str) return true;
        //return str.length()==0?true:false;
        return str.length() == 0;
    }

    /**
     * 清除数字和空格
     */
    public static String cleanBlankOrDigit(String str)
    {
        if (isBlankOrNull(str)) return "null";
        return Pattern.compile("\\d|\\s").matcher(str).replaceAll("");
    }
}


