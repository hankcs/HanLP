/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-07-15 PM2:20</create-date>
 *
 * <copyright file="ComputeAccuracy.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.word2vec;

import java.io.*;

/**
 * @author hankcs
 */
public class ComputeAccuracy
{
    final static int max_size = 2000;         // max length of strings
    final static int N = 1;                   // number of closest words
    final static int max_w = 50;              // max length of vocabulary entries

    public static void main(String[] argv) throws IOException
    {
        BufferedReader f;
        String st1 = null, st2, st3, st4;
        String[] bestw = new String[N];
        double dist, len;
        double[] bestd = new double[N];
        double[] vec = new double[max_size];
        int words = 0, size = 0, a, b, c, d, b1, b2, b3, threshold = 0;
        double M[];
        String vocab[];
        int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
        if (argv == null || argv.length != 3)
        {
            printf("Usage: ./compute-accuracy <FILE> <threshold> <QUESTION FILE>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000). Question file contains questions and answers\n");
            return;
        }
        String file_name = argv[0];
        threshold = Integer.parseInt(argv[1]);
        try
        {
            f = new BufferedReader(new InputStreamReader(new FileInputStream(file_name), "UTF-8"));
        }
        catch (FileNotFoundException e)
        {
            printf("Input file not found\n");
            System.exit(-1);
            return;
        }
        catch (UnsupportedEncodingException e)
        {
            e.printStackTrace();
            return;
        }

        try
        {
            String[] params = f.readLine().split("\\s");
            words = Integer.parseInt(params[0]);
            if (words > threshold) words = threshold;
            size = Integer.parseInt(params[1]);
            vocab = new String[words];
            M = new double[words * size];
            for (b = 0; b < words; b++)
            {
                params = f.readLine().split("\\s");
                vocab[b] = params[0].toUpperCase();
                for (a = 0; a < size; a++)
                {
                    M[a + b * size] = Double.parseDouble(params[1 + a]);
                }
                len = 0;
                for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
                len = Math.sqrt(len);
                for (a = 0; a < size; a++) M[a + b * size] /= len;
            }
            f.close();
        }
        catch (IOException e)
        {
            printf("IO error\n");
            System.exit(-2);
            return;
        }
        catch (OutOfMemoryError e)
        {
            printf("Cannot allocate memory: %lld MB\n", words * size * 8 / 1048576);
            System.exit(-3);
            return;
        }

        TCN = 0;
        BufferedReader stdin = null;
        try
        {
            stdin = new BufferedReader(new InputStreamReader(new FileInputStream(argv[2])));
        }
        catch (FileNotFoundException e)
        {
            printf("Question file %s not found\n", argv[2]);
        }
        while (true)
        {
            for (a = 0; a < N; a++) bestd[a] = 0;
            for (a = 0; a < N; a++) bestw[a] = null;
            String line = stdin.readLine();

            String[] params = null;
            if (line != null && line.length() > 0)
            {
                params = line.toUpperCase().split("\\s");
                st1 = params[0];
            }
            if (line == null || line.length() == 0 || st1.equals(":") || st1.equals("EXIT"))
            {
                if (TCN == 0) TCN = 1;
                if (QID != 0)
                {
                    printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (double) TCN * 100, CCN, TCN);
                    printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (double) TACN * 100, SEAC / (double) SECN * 100, SYAC / (double) SYCN * 100);
                }
                QID++;
                if (line == null || line.length() == 0) break;
                st1 = params[1];
                printf("%s:\n", st1);
                TCN = 0;
                CCN = 0;
                continue;
            }
            if ("EXIT".equals(st1)) break;
            st2 = params[1];
            st3 = params[2];
            st4 = params[3];
            for (b = 0; b < words; b++) if (st1.equals(vocab[b]))break;
            b1 = b;
            for (b = 0; b < words; b++) if (st2.equals(vocab[b]))break;
            b2 = b;
            for (b = 0; b < words; b++) if (st3.equals(vocab[b]))break;
            b3 = b;
            for (a = 0; a < N; a++) bestd[a] = 0;
            for (a = 0; a < N; a++) bestw[a] = null;
            TQ++;
            if (b1 == words) continue;
            if (b2 == words) continue;
            if (b3 == words) continue;
            for (b = 0; b < words; b++) if (st4.equals(vocab[b]))break;
            if (b == words) continue;
            for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
            TQS++;
            for (c = 0; c < words; c++)
            {
                if (c == b1) continue;
                if (c == b2) continue;
                if (c == b3) continue;
                dist = 0;
                for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
                for (a = 0; a < N; a++)
                {
                    if (dist > bestd[a])
                    {
                        for (d = N - 1; d > a; d--)
                        {
                            bestd[d] = bestd[d - 1];
                            bestw[d] = bestw[d - 1];
                        }
                        bestd[a] = dist;
                        bestw[a] = vocab[c];
                        break;
                    }
                }
            }
            if (st4.equals(bestw[0]))
            {
                CCN++;
                CACN++;
                if (QID <= 5) SEAC++;
                else SYAC++;
            }
            if (QID <= 5) SECN++;
            else SYCN++;
            TCN++;
            TACN++;
        }
        printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS / (double) TQ * 100);
    }

    private static void printf(String format, Object... args)
    {
        System.out.printf(format, args);
    }
}
