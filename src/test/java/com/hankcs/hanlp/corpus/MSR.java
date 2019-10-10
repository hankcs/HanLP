/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-24 10:34 AM</create-date>
 *
 * <copyright file="MSR.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.TestUtility;

/**
 * @author hankcs
 */
public class MSR
{
    public static String TRAIN_PATH = "data/test/icwb2-data/training/msr_training.utf8";
    public static String TEST_PATH = "data/test/icwb2-data/testing/msr_test.utf8";
    public static String GOLD_PATH = "data/test/icwb2-data/gold/msr_test_gold.utf8";
    public static String MODEL_PATH = "data/test/msr_cws";
    public static String OUTPUT_PATH = "data/test/msr_output.txt";
    public static String TRAIN_WORDS = "data/test/icwb2-data/gold/msr_training_words.utf8";
    public static String SIGHAN05_ROOT;

    static
    {
        SIGHAN05_ROOT = TestUtility.ensureTestData("icwb2-data", "http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip");
        if (!IOUtil.isFileExisted(TRAIN_PATH))
        {
            System.err.println("请下载 http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip 并解压为 data/test/icwb2-data");
            System.exit(1);
        }
    }
}
