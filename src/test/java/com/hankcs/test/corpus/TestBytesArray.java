/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/30 14:44</create-date>
 *
 * <copyright file="TestBytesArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ByteArrayFileStream;
import com.hankcs.hanlp.model.maxent.MaxEntModel;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;

/**
 * @author hankcs
 */
public class TestBytesArray extends TestCase
{

    public static final String DATA_OUT_DAT = "data/test/out.dat";

    public void testWriteAndRead() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_OUT_DAT));
        out.writeChar('H');
        out.writeChar('e');
        out.writeChar('l');
        out.writeChar('l');
        out.writeChar('o');
        out.close();
        ByteArray byteArray = ByteArray.createByteArray(DATA_OUT_DAT);
        while (byteArray.hasMore())
        {
            System.out.println(byteArray.nextChar());
        }
    }

    public void testWriteBigFile() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_OUT_DAT));
        for (int i = 0; i < 10000; i++)
        {
            out.writeInt(i);
        }
        out.close();
    }

    public void testStream() throws Exception
    {
        ByteArray byteArray = ByteArrayFileStream.createByteArrayFileStream(DATA_OUT_DAT);
        while (byteArray.hasMore())
        {
            System.out.println(byteArray.nextInt());
        }
    }

    /**
     * 无法在-Xms512m -Xmx512m -Xmn256m下运行<br>
     *     java.lang.OutOfMemoryError: GC overhead limit exceeded
     * @throws Exception
     */
    public void testLoadByteArray() throws Exception
    {
        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);
    }

    /**
     * 能够在-Xms512m -Xmx512m -Xmn256m下运行
     * @throws Exception
     */
    public void testLoadByteArrayStream() throws Exception
    {
        ByteArray byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);
    }

    public void testBenchmark() throws Exception
    {
        long start;

        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);

        byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);

        start = System.currentTimeMillis();
        byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);
        System.out.printf("ByteArray: %d ms\n", (System.currentTimeMillis() - start));

        start = System.currentTimeMillis();
        byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
        MaxEntModel.create(byteArray);
        System.out.printf("ByteArrayStream: %d ms\n", (System.currentTimeMillis() - start));

//        ByteArray: 2626 ms
//        ByteArrayStream: 4165 ms
    }
}
