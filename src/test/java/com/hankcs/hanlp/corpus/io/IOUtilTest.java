package com.hankcs.hanlp.corpus.io;

import junit.framework.TestCase;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.Random;

public class IOUtilTest extends TestCase
{
    public void testReadBytesFromOtherInputStream() throws Exception
    {
        Random random = new Random(System.currentTimeMillis());
        byte[] originalData = new byte[1024 * 1024]; // 1MB
        random.nextBytes(originalData);
        ByteArrayInputStream is = new ByteArrayInputStream(originalData){
            @Override
            public synchronized int available()
            {
                int realAvailable = super.available();
                if (realAvailable > 0)
                {
                    return 2048; // 模拟某些网络InputStream
                }
                return realAvailable;
            }
        };
        byte[] readData = IOUtil.readBytesFromOtherInputStream(is);
        assertEquals(originalData.length, readData.length);
        for (int i = 0; i < originalData.length; i++)
        {
            assertEquals(originalData[i], readData[i]);
        }
    }

    public void testUTF8BOM() throws Exception
    {
        File tempFile = File.createTempFile("hanlp-", ".txt");
        tempFile.deleteOnExit();
        IOUtil.saveTxt(tempFile.getAbsolutePath(), "\uFEFF第1行\n第2行");
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(tempFile.getAbsolutePath());
        int i = 1;
        for (String line : lineIterator)
        {
            assertEquals(String.format("第%d行", i++), line);
        }
    }
}