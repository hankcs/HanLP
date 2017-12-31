package com.hankcs.hanlp.mining.word2vec;

import java.io.*;
import java.util.TreeMap;

public class TextFileCorpus extends Corpus
{

    private static final int VOCAB_MAX_SIZE = 30000000;

    private int minReduce = 1;
    private BufferedReader raf = null;
    private DataOutputStream cache;

    public TextFileCorpus(Config config) throws IOException
    {
        super(config);
    }

    @Override
    public void shutdown() throws IOException
    {
        Utils.closeQuietly(raf);
        wordsBuffer = null;
    }

    @Override
    public void rewind(int numThreads, int id) throws IOException
    {
        super.rewind(numThreads, id);
    }

    @Override
    public String nextWord() throws IOException
    {
        return readWord(raf);
    }

    /**
     * Reduces the vocabulary by removing infrequent tokens
     */
    void reduceVocab()
    {
        table = new int[vocabSize];
        int j = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            if (vocab[i].cn > minReduce)
            {
                vocab[j].cn = vocab[i].cn;
                vocab[j].word = vocab[i].word;
                table[vocabIndexMap.get(vocab[j].word)] = j;
                j++;
            }
            else
            {
                table[vocabIndexMap.get(vocab[j].word)] = -4;
            }
        }
        // adjust the index in the cache
        try
        {
            cache.close();
            File fixingFile = new File(cacheFile.getAbsolutePath() + ".fixing");
            cache = new DataOutputStream(new FileOutputStream(fixingFile));
            DataInputStream oldCache = new DataInputStream(new FileInputStream(cacheFile));
            while (oldCache.available() >= 4)
            {
                int oldId = oldCache.readInt();
                if (oldId < 0)
                {
                    cache.writeInt(oldId);
                    continue;
                }
                int id = table[oldId];
                if (id == -4) continue;
                cache.writeInt(id);
            }
            oldCache.close();
            cache.close();
            if (!fixingFile.renameTo(cacheFile))
            {
                throw new RuntimeException(String.format("moving %s to %s failed", fixingFile.getAbsolutePath(), cacheFile.getName()));
            }
            cache = new DataOutputStream(new FileOutputStream(cacheFile));
        }
        catch (IOException e)
        {
            throw new RuntimeException(String.format("failed to adjust cache file", e));
        }
        table = null;
        vocabSize = j;
        vocabIndexMap.clear();
        for (int i = 0; i < vocabSize; i++)
        {
            vocabIndexMap.put(vocab[i].word, i);
        }
        minReduce++;
    }

    public void learnVocab() throws IOException
    {
        vocab = new VocabWord[vocabMaxSize];
        vocabIndexMap = new TreeMap<String, Integer>();
        vocabSize = 0;

        final File trainFile = new File(config.getInputFile());

        BufferedReader raf = null;
        FileInputStream fileInputStream = null;
        cache = null;
        vocabSize = 0;
        TrainingCallback callback = config.getCallback();
        try
        {
            fileInputStream = new FileInputStream(trainFile);
            raf = new BufferedReader(new InputStreamReader(fileInputStream, encoding));
            cacheFile = File.createTempFile(String.format("corpus_%d", System.currentTimeMillis()), ".bin");
            cache = new DataOutputStream(new FileOutputStream(cacheFile));
            while (true)
            {
                String word = readWord(raf);
                if (word == null && eoc) break;
                trainWords++;
                if (trainWords % 100000 == 0)
                {
                    if (callback == null)
                    {
                        System.err.printf("%c%.2f%% %dK", 13,
                                          (1.f - fileInputStream.available() / (float) trainFile.length()) * 100.f,
                                          trainWords / 1000);
                        System.err.flush();
                    }
                    else
                    {
                        callback.corpusLoading((1.f - fileInputStream.available() / (float) trainFile.length()) * 100.f);
                    }
                }
                int idx = searchVocab(word);
                if (idx == -1)
                {
                    idx = addWordToVocab(word);
                    vocab[idx].cn = 1;
                }
                else vocab[idx].cn++;
                if (vocabSize > VOCAB_MAX_SIZE * 0.7)
                {
                    reduceVocab();
                    idx = searchVocab(word);
                }
                cache.writeInt(idx);
            }
        }
        finally
        {
            Utils.closeQuietly(fileInputStream);
            Utils.closeQuietly(raf);
            Utils.closeQuietly(cache);
            System.err.println();
        }

        if (callback == null)
        {
            System.err.printf("%c100%% %dK", 13, trainWords / 1000);
            System.err.flush();
        }
        else
        {
            callback.corpusLoading(100);
            callback.corpusLoaded(vocabSize, trainWords, trainWords);
        }
    }

    String[] wordsBuffer = new String[0];
    int wbp = wordsBuffer.length;

    /**
     * Reads a single word from a file, assuming space + tab + EOL to be word boundaries
     *
     * @param raf
     * @return null if EOF
     * @throws IOException
     */
    String readWord(BufferedReader raf) throws IOException
    {
        while (true)
        {
            // check the buffer first
            if (wbp < wordsBuffer.length)
            {
                return wordsBuffer[wbp++];
            }

            String line = raf.readLine();
            if (line == null)
            {      // end of corpus
                eoc = true;
                return null;
            }
            line = line.trim();
            if (line.length() == 0)
            {
                continue;
            }
            cache.writeInt(-3); // mark end of sentence
            wordsBuffer = line.split("\\s+");
            wbp = 0;
            eoc = false;
        }
    }
}
