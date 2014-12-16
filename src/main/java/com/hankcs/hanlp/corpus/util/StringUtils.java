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


    /**
     * Unicode 编码并不只是为某个字符简单定义了一个编码，而且还将其进行了归类。
     * <p/>
     * /pP 其中的小写 p 是 property 的意思，表示 Unicode 属性，用于 Unicode 正表达式的前缀。
     * <p/>
     * 大写 P 表示 Unicode 字符集七个字符属性之一：标点字符。\\pP‘’“”]",如果在 JDK 5 或以下的环境中，全角单引号对、双引号对
     * <p/>
     * 其他六个是
     * L：字母；
     * M：标记符号（一般不会单独出现）；
     * Z：分隔符（比如空格、换行等）；
     * S：符号（比如数学符号、货币符号等）；
     * N：数字（比如阿拉伯数字、罗马数字等）；
     * C：其他字符
     */
    public static void main(String[] args)
    {
        System.out.println(replaceSpecialtyStr("中国电信2011年第一批IT设备集中采购-存储备份&（），)(UNIX服务器", PATTERN, ""));
    }
}


