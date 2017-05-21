package com.hankcs.hanlp.corpus.io;

import junit.framework.TestCase;

import java.io.ByteArrayInputStream;
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
}