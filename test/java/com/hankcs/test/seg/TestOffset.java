/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/11 16:03</create-date>
 *
 * <copyright file="TestOffset.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.IndexTokenizer;
import junit.framework.TestCase;

import java.util.List;

/**
 * @author hankcs
 */
public class TestOffset extends TestCase
{
    public void testOffset() throws Exception
    {
        String text = "中华人民共和国在哪里";
        for (int i = 0; i < text.length(); ++i)
        {
            System.out.print(text.charAt(i) + "" + i + " ");
        }
        System.out.println();
        List<Term> termList = IndexTokenizer.segment(text);
        for (Term term : termList)
        {
            System.out.println(term.word + " " + term.offset + " " + (term.offset + term.length()));
        }
    }
}
