package com.hankcs.hanlp.mining.word2vec;

import java.io.*;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.List;

public final class KMeansClustering
{
    static final Charset ENCODING = Charset.forName("UTF-8");
    private final VectorsReader reader;
    private final int clcn;
    private final String outFile;

    public KMeansClustering(VectorsReader reader, int k, String outFile)
    {
        this.reader = reader;
        this.clcn = k;
        this.outFile = outFile;
    }

    public void clustering() throws IOException
    {
        final int vocabSize = reader.getNumWords();
        final int layer1Size = reader.getSize();

        OutputStream os = null;
        Writer w = null;
        PrintWriter pw = null;

        try
        {
            os = new FileOutputStream(outFile);
            w = new OutputStreamWriter(os, ENCODING);
            pw = new PrintWriter(w);

            // Run K-means on the word vectors
            System.err.printf("now computing K-means clustering (K=%d)\n", clcn);
            final int MAX_ITER = 10;
            final int[] centcn = new int[clcn];
            final int[] cl = new int[vocabSize];
            final int centSize = clcn * layer1Size;
            final double[] cent = new double[centSize];

            for (int i = 0; i < vocabSize; i++)
                cl[i] = i % clcn;

            for (int it = 0; it < MAX_ITER; it++)
            {
                for (int j = 0; j < centSize; j++)
                    cent[j] = 0;
                for (int j = 0; j < clcn; j++)
                    centcn[j] = 1;
                for (int k = 0; k < vocabSize; k++)
                {
                    for (int l = 0; l < layer1Size; l++)
                    {
                        cent[layer1Size * cl[k] + l] += reader.getMatrixElement(k, l);
                    }
                    centcn[cl[k]]++;
                }
                for (int j = 0; j < clcn; j++)
                {
                    double closev = 0;
                    for (int k = 0; k < layer1Size; k++)
                    {
                        cent[layer1Size * j + k] /= centcn[j];
                        closev += cent[layer1Size * j + k] * cent[layer1Size * j + k];
                    }
                    closev = Math.sqrt(closev);
                    for (int k = 0; k < layer1Size; k++)
                    {
                        cent[layer1Size * j + k] /= closev;
                    }
                }
                for (int k = 0; k < vocabSize; k++)
                {
                    double closev = -10;
                    int closeid = 0;
                    for (int l = 0; l < clcn; l++)
                    {
                        double x = 0;
                        for (int j = 0; j < layer1Size; j++)
                        {
                            x += cent[layer1Size * l + j] * reader.getMatrixElement(k, j);
                        }
                        if (x > closev)
                        {
                            closev = x;
                            closeid = l;
                        }
                    }
                    cl[k] = closeid;
                }
            }
            // Save the K-means classes
            System.err.printf("now saving the result of K-means clustering to the file %s\n", outFile);
            List<String>[] cluster = new List[clcn];
            for (int i = 0; i < cluster.length; i++)
            {
                cluster[i] = new LinkedList<String>();
            }
            for (int i = 0; i < vocabSize; i++)
            {
                cluster[cl[i]].add(reader.getWord(i));
            }
            for (int i = 0; i < cluster.length; i++)
            {
                for (String word : cluster[i])
                {
                    pw.printf("%s\t%d\n", word, i);
                }
            }
        }
        finally
        {
            Utility.closeQuietly(pw);
            Utility.closeQuietly(w);
            Utility.closeQuietly(os);
        }
    }
}
