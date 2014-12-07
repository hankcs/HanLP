package com.hankcs.test.algorithm;

import com.hankcs.hanlp.corpus.io.ByteArray;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;

public class ByteUtilTest extends TestCase
{

    public static final String DATA_TEST_OUT_BIN = "data/test/out.bin";

    public void testReadDouble() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        out.writeDouble(0.123456789);
        out.writeInt(3389);
        ByteArray byteArray = ByteArray.createByteArray(DATA_TEST_OUT_BIN);
        System.out.println(byteArray.nextDouble());
        System.out.println(byteArray.nextInt());
    }
}