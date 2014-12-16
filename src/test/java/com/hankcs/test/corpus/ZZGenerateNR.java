/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 14:35</create-date>
 *
 * <copyright file="ZZGenerateNR.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

/**
 * @author hankcs
 */
public class ZZGenerateNR
{
    public static void main(String[] strings)
    {
        String text = "B\tPf\t姓氏\t张华平先生\n" +
                "C\tPm\t双名的首字\t张华平先生\n" +
                "D\tPt\t双名的末字\t张华平先生\n" +
                "E\tPs\t单名\t张浩说：“我是一个好人” \n" +
                "F\tPpf\t前缀\t老刘、小李 \n" +
                "G\tPlf\t后缀\t王总、刘老、肖氏、吴妈、叶帅\n" +
                "K\tPp\t人名的上文\t又来到于洪洋的家。\n" +
                "L\tPn\t人名的下文\t新华社记者黄文摄\n" +
                "M\tPpn\t两个中国人名之间的成分\t编剧邵钧林和稽道青说\n" +
                "U\tPpf\t人名的上文和姓成词\t这里有关天培的壮烈\n" +
                "V\tPnw\t人名的末字和下文成词\t龚学平等领导, 邓颖超生前\n" +
                "X\tPfm\t姓与双名的首字成词\t王国维、\n" +
                "Y\tPfs\t姓与单名成词\t高峰、汪洋\n" +
                "Z\tPmt\t双名本身成词\t张朝阳\n" +
                "A\tPo\t以上之外其他的角色\t\n";

        for (String line : text.split("\n"))
        {
            System.out.printf("/**\n* %s\n*/\n%s,\n\n", line.substring(1, line.length()), line.substring(0, 1));
        }
    }
}
