/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/20 23:40</create-date>
 *
 * <copyright file="MDAGUnsimplifyTest.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.algorithm;

import com.hankcs.hanlp.collection.MDAG.MDAG;
import com.hankcs.hanlp.corpus.io.ByteArray;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;

/**
 * 希望在此测试解压缩
 * @author hankcs
 */
public class MDAGUnsimplifyTest extends TestCase
{
    private static final String DATA_TEST_OUT_BIN = "data/test/out.bin";

    public void testSimplify() throws Exception
    {
        MDAG mdag = new MDAG();
        mdag.addString("hers");
        mdag.addString("his");
        mdag.addString("she");
        mdag.addString("he");
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        mdag.save(out);

        mdag = new MDAG();
        mdag.load(ByteArray.createByteArray(DATA_TEST_OUT_BIN));
        System.out.println(mdag.contains("his"));
    }
}
