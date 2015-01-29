/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/1/29 17:42</create-date>
 *
 * <copyright file="TestLDA.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.io.File;
import java.util.List;

/**
 * @author hankcs
 */
public class TestLDA extends TestCase
{
    public void testSegmentCorpus() throws Exception
    {
        File root = new File("D:\\Doc\\语料库\\搜狗文本分类语料库精简版");
        for (File folder : root.listFiles())
        {
            if (folder.isDirectory())
            {
                for (File file : folder.listFiles())
                {
                    System.out.println(file.getAbsolutePath());
                    List<Term> termList = HanLP.segment(IOUtil.readTxt(file.getAbsolutePath()));
                    StringBuilder sbOut = new StringBuilder();
                    for (Term term : termList)
                    {
                        if (CoreStopWordDictionary.shouldInclude(term))
                        {
                            sbOut.append(term.word).append(" ");
                        }
                    }
                    IOUtil.saveTxt("D:\\Doc\\语料库\\segmented\\" + folder.getName() + "_" + file.getName(), sbOut.toString());
                }
            }
        }
    }
}
