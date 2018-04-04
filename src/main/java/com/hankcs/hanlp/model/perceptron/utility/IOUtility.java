/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM7:29</create-date>
 *
 * <copyright file="IOUtility.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.utility;

import com.hankcs.hanlp.classification.utilities.io.ConsoleLogger;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;

import java.io.*;
import java.util.regex.Pattern;


/**
 * @author hankcs
 */
public class IOUtility extends IOUtil
{
    private static Pattern PATTERN_SPACE = Pattern.compile("\\s+");

    public static String[] readLineToArray(String line)
    {
        line = line.trim();
        if (line.length() == 0) return new String[0];
        return PATTERN_SPACE.split(line);
    }

    public static int loadInstance(final String path, InstanceHandler handler) throws IOException
    {
        ConsoleLogger logger = new ConsoleLogger();
        int size = 0;
        File root = new File(path);
        File allFiles[];
        if (root.isDirectory())
        {
            allFiles = root.listFiles(new FileFilter()
            {
                @Override
                public boolean accept(File pathname)
                {
                    return pathname.isFile() && pathname.getName().endsWith(".txt");
                }
            });
        }
        else
        {
            allFiles = new File[]{root};
        }

        for (File file : allFiles)
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = br.readLine()) != null)
            {
                line = line.trim();
                if (line.length() == 0)
                {
                    continue;
                }
                Sentence sentence = Sentence.create(line);
                if (sentence.wordList.size() == 0) continue;
                ++size;
                if (size % 1000 == 0)
                {
                    logger.err("%c语料: %dk...", 13, size / 1000);
                }
                // debug
//                if (size == 100) break;
                if (handler.process(sentence)) break;
            }
        }

        return size;
    }

    public static double[] evaluate(Instance[] instances, LinearModel model)
    {
        int[] stat = new int[2];
        for (int i = 0; i < instances.length; i++)
        {
            evaluate(instances[i], model, stat);
            if (i % 100 == 0 || i == instances.length - 1)
            {
                System.err.printf("%c进度: %.2f%%", 13, (i + 1) / (float) instances.length * 100);
                System.err.flush();
            }
        }
        return new double[]{stat[1] / (double) stat[0] * 100};
    }

    public static void evaluate(Instance instance, LinearModel model, int[] stat)
    {
        int[] predLabel = new int[instance.length()];
        model.viterbiDecode(instance, predLabel);
        stat[0] += instance.tagArray.length;
        for (int i = 0; i < predLabel.length; i++)
        {
            if (predLabel[i] == instance.tagArray[i])
            {
                ++stat[1];
            }
        }
    }
}
