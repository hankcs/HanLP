/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/25 19:33</create-date>
 *
 * <copyright file="CheckDateElements.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.seg.NShort.Segment;

import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class testCheckDateElements
{
    public static void main(String[] args)
    {
        List<List<WordResult>> wordResults = new LinkedList<>();
        wordResults.add(Segment.parse("3-4月"));
        wordResults.add(Segment.parse("3-4月份"));
        wordResults.add(Segment.parse("3-4季"));
        wordResults.add(Segment.parse("3-4年"));
        wordResults.add(Segment.parse("3-4人"));
        wordResults.add(Segment.parse("2014年"));
        wordResults.add(Segment.parse("04年"));
        wordResults.add(Segment.parse("12点半"));
        wordResults.add(Segment.parse("1.abc"));

        for (List<WordResult> result : wordResults)
        {
            System.out.println(result);
        }
    }
}
