/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/29 21:11</create-date>
 *
 * <copyright file="TestICWB.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

/**
 * 玩玩ICWB的数据
 * @author hankcs
 */
public class TestICWB extends TestCase
{

    public static final String PATH = "D:\\Doc\\语料库\\icwb2-data\\training\\msr_training.utf8";

    public void testGenerateBMES() throws Exception
    {
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(PATH +  ".bmes.txt")));
        for (String line : IOUtil.readLineListWithLessMemory(PATH))
        {
            String[] wordArray = line.split("\\s");
            for (String word : wordArray)
            {
                if (word.length() == 1)
                {
                    bw.write(word + "_S ");
                }
                else if (word.length() > 1)
                {
                    bw.write(word.charAt(0) + "_B ");
                    for (int i = 1; i < word.length() - 1; ++i)
                    {
                        bw.write(word.charAt(i) + "_M ");
                    }
                    bw.write(word.charAt(word.length() - 1) + "_E ");
                }
            }
            bw.newLine();
        }
        bw.close();
    }
}
