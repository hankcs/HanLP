package com.hankcs.hanlp.collection.trie.datrie;

import com.hankcs.hanlp.corpus.io.ByteArray;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;

public class MutableDoubleArrayTrieIntegerTest extends TestCase
{
    MutableDoubleArrayTrieInteger mdat;
    private int size;

    @Override
    public void setUp() throws Exception
    {
        mdat = new MutableDoubleArrayTrieInteger();
        size = 64;
        for (int i = 0; i < size; ++i)
        {
            mdat.put(String.valueOf(i), i);
        }
    }

    public void testSaveLoad() throws Exception
    {
        File tempFile = File.createTempFile("hanlp", ".mdat");
        mdat.save(new DataOutputStream(new FileOutputStream(tempFile)));
        mdat = new MutableDoubleArrayTrieInteger();
        mdat.load(ByteArray.createByteArray(tempFile.getAbsolutePath()));
        assertEquals(size, mdat.size());
        for (int i = 0; i < size; ++i)
        {
            assertEquals(i, mdat.get(String.valueOf(i)));
        }

        for (int i = size; i < 2 * size; ++i)
        {
            assertEquals(-1, mdat.get(String.valueOf(i)));
        }
    }
}