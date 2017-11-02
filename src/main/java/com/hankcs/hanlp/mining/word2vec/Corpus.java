package com.hankcs.hanlp.mining.word2vec;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

public abstract class Corpus
{

    protected File cacheFile;
    protected Config config;
    protected int trainWords = 0;
    protected int vocabSize;
    protected int vocabMaxSize = 1000;
    protected VocabWord[] vocab;
    protected Map<String, Integer> vocabIndexMap;
    protected boolean eoc = true;    // end of corpus
    protected Charset encoding = Charset.forName("UTF-8");
    protected int[] table;

    public Corpus(Config config) throws IOException
    {
        this.config = config;
    }

    public Corpus(Corpus cloneSrc) throws IOException
    {
        trainWords = cloneSrc.trainWords;
        vocabSize = cloneSrc.vocabSize;
        vocab = cloneSrc.vocab;
        vocabIndexMap = cloneSrc.vocabIndexMap;
        table = cloneSrc.table;
    }

    public boolean endOfCorpus()
    {
        return eoc;
    }

    /**
     * Adds a word to the vocabulary
     *
     * @param word
     * @return
     */
    protected int addWordToVocab(String word)
    {
        vocab[vocabSize] = new VocabWord(word);
        vocabSize++;

        // Reallocate memory if needed
        if (vocabSize + 2 >= vocabMaxSize)
        {
            vocabMaxSize += 1000;
            VocabWord[] temp = new VocabWord[vocabMaxSize];
            System.arraycopy(vocab, 0, temp, 0, vocabSize);
            vocab = temp;
        }
        vocabIndexMap.put(word, vocabSize - 1);
        return vocabSize - 1;
    }

    public int getTrainWords()
    {
        return trainWords;
    }

    public int getVocabSize()
    {
        return vocabSize;
    }

    public VocabWord[] getVocab()
    {
        return vocab;
    }

    public Map<String, Integer> getVocabIndexMap()
    {
        return vocabIndexMap;
    }

    /**
     * reset current corpus to initial status
     */
    public void rewind(int numThreads, int id) throws IOException
    {
        eoc = false;
    }

    /**
     * @return -4 if is dropped, -3 if end of sentence, -2 if end of corpus, -1 if word not found or index value of the word
     * @throws IOException
     */
    public int readWordIndex() throws IOException
    {
        String word = nextWord();
        if (word == null)
        {
            if (eoc) return -2;    // end of corpus
            else return -3;       // end of sentence
        }
        else
        {
            return searchVocab(word);     // index value of the word
        }
    }

    /**
     * Read the next word from the corpus
     *
     * @return next word that is read from the corpus. null will be returned
     * @throws IOException
     */
    public abstract String nextWord() throws IOException;

    /**
     * Close the corpus and it cannot be read any more.
     */
    public void close() throws IOException
    {
        shutdown();
        cacheFile.delete(); // remove rubbish
    }

    public void shutdown() throws IOException
    {
        table = null;
    }

    /**
     * Returns position of a word in the vocabulary; if the word is not found, returns -1
     *
     * @param word
     * @return
     */
    int searchVocab(String word)
    {
        if (word == null) return -1;
        Integer pos = vocabIndexMap.get(word);
        return pos == null ? -1 : pos.intValue();
    }

    /**
     * Sorts the vocabulary by frequency using word counts
     */
    void sortVocab()
    {
        Arrays.sort(vocab, 0, vocabSize);

        // re-build vocabIndexMap
        final int size = vocabSize;
        trainWords = 0;
        table = new int[size];
        for (int i = 0; i < size; i++)
        {
            VocabWord word = vocab[i];
            // Words occuring less than min_count times will be discarded from the vocab
            if (word.cn < config.getMinCount())
            {
                table[vocabIndexMap.get(word.word)] = -4;
                vocabSize--;
            }
            else
            {
                // Hash will be re-computed, as after the sorting it is not actual
                table[vocabIndexMap.get(word.word)] = i;
                setVocabIndexMap(word, i);
            }
        }
        // lose weight
        vocabIndexMap = null;

        VocabWord[] nvocab = new VocabWord[vocabSize];
        System.arraycopy(vocab, 0, nvocab, 0, vocabSize);

    }

    void setVocabIndexMap(VocabWord src, int pos)
    {
//        vocabIndexMap.put(src.word, pos);
        trainWords += src.cn;
    }

    /**
     * Create binary Huffman tree using the word counts.
     * Frequent words will have short uniqe binary codes
     */
    void createBinaryTree()
    {
        int[] point = new int[VocabWord.MAX_CODE_LENGTH];
        char[] code = new char[VocabWord.MAX_CODE_LENGTH];
        int[] count = new int[vocabSize * 2 + 1];
        char[] binary = new char[vocabSize * 2 + 1];
        int[] parentNode = new int[vocabSize * 2 + 1];

        for (int i = 0; i < vocabSize; i++)
            count[i] = vocab[i].cn;
        for (int i = vocabSize; i < vocabSize * 2; i++)
            count[i] = Integer.MAX_VALUE;
        int pos1 = vocabSize - 1;
        int pos2 = vocabSize;
        // Following algorithm constructs the Huffman tree by adding one node at a time
        int min1i, min2i;
        for (int i = 0; i < vocabSize - 1; i++)
        {
            // First, find two smallest nodes 'min1, min2'
            if (pos1 >= 0)
            {
                if (count[pos1] < count[pos2])
                {
                    min1i = pos1;
                    pos1--;
                }
                else
                {
                    min1i = pos2;
                    pos2++;
                }
            }
            else
            {
                min1i = pos2;
                pos2++;
            }
            if (pos1 >= 0)
            {
                if (count[pos1] < count[pos2])
                {
                    min2i = pos1;
                    pos1--;
                }
                else
                {
                    min2i = pos2;
                    pos2++;
                }
            }
            else
            {
                min2i = pos2;
                pos2++;
            }
            count[vocabSize + i] = count[min1i] + count[min2i];
            parentNode[min1i] = vocabSize + i;
            parentNode[min2i] = vocabSize + i;
            binary[min2i] = 1;
        }
        // Now assign binary code to each vocabulary word
        for (int j = 0; j < vocabSize; j++)
        {
            int k = j;
            int i = 0;
            while (true)
            {
                code[i] = binary[k];
                point[i] = k;
                i++;
                k = parentNode[k];
                if (k == vocabSize * 2 - 2) break;
            }
            vocab[j].codelen = i;
            vocab[j].point[0] = vocabSize - 2;
            for (k = 0; k < i; k++)
            {
                vocab[j].code[i - k - 1] = code[k];
                vocab[j].point[i - k] = point[k] - vocabSize;
            }
        }
    }
}
