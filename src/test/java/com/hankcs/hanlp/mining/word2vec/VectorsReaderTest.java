package com.hankcs.hanlp.mining.word2vec;

import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.File;

public class VectorsReaderTest extends TestCase
{
    public void testReadVectorFile() throws Exception
    {
        File tempFile = File.createTempFile("hanlp-vector", ".txt");
        tempFile.deleteOnExit();
        BufferedWriter bw = IOUtil.newBufferedWriter(tempFile.getAbsolutePath());
        bw.write("3 1\n" +
                     "cat 1.1\n" +
                     " 2.2\n" +
                     "dog 3.3\n"
        );
        bw.close();

        VectorsReader reader = new VectorsReader(tempFile.getAbsolutePath());
        reader.readVectorFile();
        assertEquals(2, reader.words);
        assertEquals(2, reader.vocab.length);
        assertEquals(2, reader.matrix.length);
        assertEquals(1f, reader.matrix[1][0]);
    }
}