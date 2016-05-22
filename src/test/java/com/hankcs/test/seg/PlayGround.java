/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 12:55</create-date>
 *
 * <copyright file="TestPlayGround.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import junit.framework.TestCase;

import java.util.List;
import java.util.Scanner;

/**
 * @author hankcs
 */
public class PlayGround extends TestCase
{
    public static void main(String[] args)
    {
        Scanner scanner = new Scanner(System.in);
        String line;
        while ((line = scanner.nextLine()).length() > 0
                )
        {
            seg(line);
        }
    }

    private static void seg(String sentence)
    {
        List<Term> terms = NShortSegment.parse(sentence);
        for (Term wr : terms)
        {
            System.out.print(wr.word + wr.nature);
        }
        System.out.println();
    }

    public void testCharSequence() throws Exception
    {
        CharSequence s = "hello";

    }
}
