/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-04 5:36 PM</create-date>
 *
 * <copyright file="PKU.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.TestUtility;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * @author hankcs
 */
public class PKU
{
    public static String PKU199801;
    public static String PKU199801_TRAIN = "data/test/pku98/199801-train.txt";
    public static String PKU199801_TEST = "data/test/pku98/199801-test.txt";
    public static String POS_MODEL = "/pos.bin";
    public static final String PKU_98 = TestUtility.ensureTestData("pku98", "http://hanlp.linrunsoft.com/release/corpus/pku98.zip");

    static
    {
        PKU199801 = PKU_98 + "/199801.txt";
        POS_MODEL = PKU_98 + POS_MODEL;
        if (!IOUtil.isFileExisted(PKU199801_TRAIN))
        {
            ArrayList<String> all = new ArrayList<String>();
            IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(PKU199801);
            while (lineIterator.hasNext())
            {
                all.add(lineIterator.next());
            }
            try
            {
                BufferedWriter bw = IOUtil.newBufferedWriter(PKU199801_TRAIN);
                for (String line : all.subList(0, (int) (all.size() * 0.9)))
                {
                    bw.write(line);
                    bw.newLine();
                }
                bw.close();

                bw = IOUtil.newBufferedWriter(PKU199801_TEST);
                for (String line : all.subList((int) (all.size() * 0.9), all.size()))
                {
                    bw.write(line);
                    bw.newLine();
                }
                bw.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }
}
