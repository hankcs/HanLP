package com.hankcs.hanlp.corpus.io;

import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

public class ByteArrayTest extends TestCase
{
    public void testNextBoolean() throws Exception
    {
        File tempFile = File.createTempFile("hanlp-", ".dat");
        DataOutputStream out = new DataOutputStream(new FileOutputStream(tempFile));
        out.writeBoolean(true);
        out.writeBoolean(false);
        ByteArray byteArray = ByteArray.createByteArray(tempFile.getAbsolutePath());
        assertNotNull(byteArray);
        assertEquals(byteArray.nextBoolean(), true);
        assertEquals(byteArray.nextBoolean(), false);
        tempFile.deleteOnExit();
    }
}