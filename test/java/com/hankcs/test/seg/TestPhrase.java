/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/9 16:43</create-date>
 *
 * <copyright file="TestPhrase.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.corpus.io.FolderWalker;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.phrase.MutualInformationEntropyPhraseExtractor;
import com.hankcs.hanlp.tokenizer.StandTokenizer;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * @author hankcs
 */
public class TestPhrase extends TestCase
{
    static final String FOLDER = "D:\\Doc\\语料库\\上海静安\\";
    public void testExtract() throws Exception
    {
        List<File> fileList = FolderWalker.open(FOLDER);
        Set<String> phraseSet = new TreeSet<>();
        int i = 0;
        for (File file : fileList)
        {
            System.out.print(++i + " / " + fileList.size() + " " + file.getName() + " ");
            String path = file.getAbsolutePath();
            List<String> phraseList = MutualInformationEntropyPhraseExtractor.extract(IOUtil.readTxt(path), 3);
            System.out.print(phraseList);
            phraseSet.addAll(phraseList);
            System.out.println();
        }
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/phrase.txt")));
        for (String phrase : phraseSet)
        {
            bw.write(phrase);
            bw.newLine();
        }
        bw.close();
    }

    public void testSeg() throws Exception
    {
        System.out.println(StandTokenizer.parse(IOUtil.readTxt(FOLDER + "南西社区暑期学生活动简讯  2010年第1期.txt")));
    }
}
