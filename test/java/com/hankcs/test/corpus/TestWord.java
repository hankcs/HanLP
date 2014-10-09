/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 17:34</create-date>
 *
 * <copyright file="TestWord.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestWord extends TestCase
{
    public void testCreate() throws Exception
    {
        assertEquals("人民网/nz", Word.create("人民网/nz").toString());
        assertEquals("[纽约/nsf 时报/n]/nz", CompoundWord.create("[纽约/nsf 时报/n]/nz").toString());
    }

    public void testSpace() throws Exception
    {
        CompoundWord compoundWord = CompoundWord.create("[9/m  11/m 后/f]/mq");
        System.out.println(compoundWord);
    }
}
