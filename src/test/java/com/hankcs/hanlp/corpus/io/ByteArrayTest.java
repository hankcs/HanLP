package com.hankcs.hanlp.corpus.io;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.maxent.MaxEntModel;
import com.hankcs.hanlp.utility.ByteUtil;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

public class ByteArrayTest extends TestCase
{
    static String DATA_TEST_OUT_BIN;
    private File tempFile;

    @Override
    public void setUp() throws Exception
    {
        tempFile = File.createTempFile("hanlp-", ".dat");
        DATA_TEST_OUT_BIN = tempFile.getAbsolutePath();
    }

    public void testReadDouble() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        double d = 0.123456789;
        out.writeDouble(d);
        int i = 3389;
        out.writeInt(i);
        ByteArray byteArray = ByteArray.createByteArray(DATA_TEST_OUT_BIN);
        assertEquals(d, byteArray.nextDouble());
        assertEquals(i, byteArray.nextInt());
    }

    public void testReadUTF() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        String utf = "hankcs你好123";
        out.writeUTF(utf);
        ByteArray byteArray = ByteArray.createByteArray(DATA_TEST_OUT_BIN);
        assertEquals(utf, byteArray.nextUTF());
    }

    public void testReadUnsignedShort() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        int utflen = 123;
        out.writeByte((byte) ((utflen >>> 8) & 0xFF));
        out.writeByte((byte) ((utflen >>> 0) & 0xFF));
        ByteArray byteArray = ByteArray.createByteArray(DATA_TEST_OUT_BIN);
        assertEquals(utflen, byteArray.nextUnsignedShort());
    }

    public void testConvertCharToInt() throws Exception
    {
//        for (int i = 0; i < Integer.MAX_VALUE; ++i)
        for (int i = 0; i < 1024; ++i)
        {
            int n = i;
            char[] twoChar = ByteUtil.convertIntToTwoChar(n);
            assertEquals(n, ByteUtil.convertTwoCharToInt(twoChar[0], twoChar[1]));
        }
    }

    public void testNextBoolean() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(tempFile));
        out.writeBoolean(true);
        out.writeBoolean(false);
        ByteArray byteArray = ByteArray.createByteArray(tempFile.getAbsolutePath());
        assertNotNull(byteArray);
        assertEquals(byteArray.nextBoolean(), true);
        assertEquals(byteArray.nextBoolean(), false);
        tempFile.deleteOnExit();
    }

    public void testWriteAndRead() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        out.writeChar('H');
        out.writeChar('e');
        out.writeChar('l');
        out.writeChar('l');
        out.writeChar('o');
        out.close();
        ByteArray byteArray = ByteArray.createByteArray(DATA_TEST_OUT_BIN);
        while (byteArray.hasMore())
        {
            byteArray.nextChar();
//            System.out.println(byteArray.nextChar());
        }
    }

    public void testWriteBigFile() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_TEST_OUT_BIN));
        for (int i = 0; i < 10000; i++)
        {
            out.writeInt(i);
        }
        out.close();
    }

    public void testStream() throws Exception
    {
        ByteArray byteArray = ByteArrayFileStream.createByteArrayFileStream(DATA_TEST_OUT_BIN);
        while (byteArray.hasMore())
        {
            System.out.println(byteArray.nextInt());
        }
    }

//    /**
//     * 无法在-Xms512m -Xmx512m -Xmn256m下运行<br>
//     *     java.lang.OutOfMemoryError: GC overhead limit exceeded
//     * @throws Exception
//     */
//    public void testLoadByteArray() throws Exception
//    {
//        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//    }
//
//    /**
//     * 能够在-Xms512m -Xmx512m -Xmn256m下运行
//     * @throws Exception
//     */
//    public void testLoadByteArrayStream() throws Exception
//    {
//        ByteArray byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//    }
//
//    public void testBenchmark() throws Exception
//    {
//        long start;
//
//        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//
//        byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//
//        start = System.currentTimeMillis();
//        byteArray = ByteArray.createByteArray(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//        System.out.printf("ByteArray: %d ms\n", (System.currentTimeMillis() - start));
//
//        start = System.currentTimeMillis();
//        byteArray = ByteArrayFileStream.createByteArrayFileStream(HanLP.Config.MaxEntModelPath + Predefine.BIN_EXT);
//        MaxEntModel.create(byteArray);
//        System.out.printf("ByteArrayStream: %d ms\n", (System.currentTimeMillis() - start));
//
////        ByteArray: 2626 ms
////        ByteArrayStream: 4165 ms
//    }
}