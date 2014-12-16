/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/29 15:23</create-date>
 *
 * <copyright file="TestPersonRecognition.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.corpus.io.FolderWalker;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.SentencesUtil;
import junit.framework.TestCase;

import java.io.File;
import java.util.List;

/**
 * @author hankcs
 */
public class TestPersonRecognition extends TestCase
{
    static final String FOLDER = "D:\\Doc\\语料库\\上海静安\\";

    public void testBatch() throws Exception
    {
        List<File> fileList = FolderWalker.open(FOLDER);
        int i = 0;
        for (File file : fileList)
        {
            System.out.println(++i + " / " + fileList.size() + " " + file.getName() + " ");
            String path = file.getAbsolutePath();
            String content = IOUtil.readTxt(path);
            DijkstraSegment segment = new DijkstraSegment();
            List<List<Term>> sentenceList = segment.seg2sentence(content);
            for (List<Term> sentence : sentenceList)
            {
                if (SentencesUtil.hasNature(sentence, Nature.nr))
                {
                    System.out.println(sentence);
                }
            }
        }
    }

    public void testNameRecognition() throws Exception
    {
        NShortSegment segment = new NShortSegment();
        System.out.println(segment.seg("华健康"));
    }
}
