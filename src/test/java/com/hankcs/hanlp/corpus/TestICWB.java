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
package com.hankcs.hanlp.corpus;

import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.LinkedList;
import java.util.List;

/**
 * 玩玩ICWB的数据
 *
 * @author hankcs
 */
public class TestICWB extends TestCase
{

//    public static final String PATH = "D:\\Doc\\语料库\\icwb2-data\\training\\msr_training.utf8";
//
//    public void testGenerateBMES() throws Exception
//    {
//        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(PATH + ".bmes.txt")));
//        for (String line : IOUtil.readLineListWithLessMemory(PATH))
//        {
//            String[] wordArray = line.split("\\s");
//            for (String word : wordArray)
//            {
//                if (word.length() == 1)
//                {
//                    bw.write(word + "\tS\n");
//                }
//                else if (word.length() > 1)
//                {
//                    bw.write(word.charAt(0) + "\tB\n");
//                    for (int i = 1; i < word.length() - 1; ++i)
//                    {
//                        bw.write(word.charAt(i) + "\tM\n");
//                    }
//                    bw.write(word.charAt(word.length() - 1) + "\tE\n");
//                }
//            }
//            bw.newLine();
//        }
//        bw.close();
//    }
//
//    public void testDumpPeople2014ToBEMS() throws Exception
//    {
//        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Tools\\CRF++-0.58\\example\\seg_cn\\2014.txt")));
//        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
//        {
//            @Override
//            public void handle(Document document)
//            {
//                List<List<Word>> simpleSentenceList = document.getSimpleSentenceList();
//                for (List<Word> wordList : simpleSentenceList)
//                {
//                    try
//                    {
//                        for (Word word : wordList)
//                        {
//
//                            bw.write(word.value);
//                            bw.write(' ');
//
//                        }
//                        bw.newLine();
//                    }
//                    catch (IOException e)
//                    {
//                        e.printStackTrace();
//                    }
//                }
//            }
//        });
//        bw.close();
//    }
}
