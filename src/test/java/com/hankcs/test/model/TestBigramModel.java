/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/11 17:44</create-date>
 *
 * <copyright file="TestBigramModel.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.model;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.bigram.BigramDependencyModel;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestBigramModel extends TestCase
{
    public void testLoad() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(BigramDependencyModel.get("传", "v", "角落", "n"));
    }
}
