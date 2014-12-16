/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 18:32</create-date>
 *
 * <copyright file="TestSentence.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.WordFactory;
import junit.framework.TestCase;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author hankcs
 */
public class TestSentence extends TestCase
{
    public void testCreate() throws Exception
    {
        String text = "人民网/nz 1月1日/t 讯/ng 据/p 《/w [纽约/nsf 时报/n]/nz 》/w 报道/v ，/w";
        Pattern pattern = Pattern.compile("(\\[(.+/[a-z]+)]/[a-z]+)|([^\\s]+/[a-z]+)");
        Matcher matcher = pattern.matcher(text);
        while (matcher.find())
        {
            String param = matcher.group();
            assertEquals(param, WordFactory.create(param).toString());
        }
        assertEquals(text, Sentence.create(text).toString());
    }
}
