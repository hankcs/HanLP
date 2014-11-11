/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/28 11:51</create-date>
 *
 * <copyright file="TestIOUtil.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.corpus.io.IOUtil;

/**
 * 测试IO
 * @author hankcs
 */
public class TestIOUtil
{
    public void testSaveTxt() throws Exception
    {
        String path = "data/out.txt";
        String content = "你好123\nabc";
        System.out.println(IOUtil.saveTxt(path, content));
    }
}
