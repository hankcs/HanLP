package com.hankcs.hanlp.mining.word2vec;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * load corpus from disk cache
 *
 * @author hankcs
 */
public class CacheCorpus extends Corpus
{
    private RandomAccessFile raf;

    public CacheCorpus(Corpus cloneSrc) throws IOException
    {
        super(cloneSrc);
        raf = new RandomAccessFile(((TextFileCorpus) cloneSrc).cacheFile, "r");
    }

    @Override
    public String nextWord() throws IOException
    {
        return null;
    }

    @Override
    public int readWordIndex() throws IOException
    {
        int id = nextId();
        while (id == -4)
        {
            id = nextId();
        }
        return id;
    }

    private int nextId() throws IOException
    {
        if (raf.length() - raf.getFilePointer() >= 4)
        {
            int id = raf.readInt();
            return id < 0 ? id : table[id];
        }

        return -2;
    }

    @Override
    public void rewind(int numThreads, int id) throws IOException
    {
        super.rewind(numThreads, id);
        raf.seek(raf.length() / 4 / numThreads * id * 4);   // spilt by id, not by bytes
    }
}