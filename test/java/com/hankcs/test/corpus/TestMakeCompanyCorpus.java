/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/18 19:48</create-date>
 *
 * <copyright file="TestMakeCompanyCorpus.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.NTDictionaryMaker;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.io.*;
import java.util.List;


/**
 * @author hankcs
 */
public class TestMakeCompanyCorpus extends TestCase
{
    public void testMake() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        String line = null;
        BufferedReader bw = new BufferedReader(new InputStreamReader(new FileInputStream("D:\\Doc\\语料库\\company.dic")));
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/test/nt/company.txt")));
        int limit = Integer.MAX_VALUE;
        while ((line = bw.readLine()) != null && limit-- > 0)
        {
            if (line.endsWith("）")) continue;
            if (line.length() < 4) continue;
            if (line.contains("个体") || line.contains("个人"))
            {
                continue;
            }
            List<Term> termList = segment.seg(line);
            if (termList.size() == 0) continue;
            Term last = termList.get(termList.size() - 1);
            last.nature = Nature.nis;
            br.write("[");
            for (Term term : termList)
            {
                br.write(term.toString());
                if (term != last) br.write(" ");
            }
            br.write("]/ntc");
            br.newLine();
            br.flush();
        }
        bw.close();
        br.close();
    }

    public void testParse() throws Exception
    {
        EasyDictionary dictionary = EasyDictionary.create("data/dictionary/2014_dictionary.txt");
        final NTDictionaryMaker nsDictionaryMaker = new NTDictionaryMaker(dictionary);
        // CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014\\", new CorpusLoader.Handler()
        CorpusLoader.walk("data/test/nt/part/", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                nsDictionaryMaker.compute(document.getComplexSentenceList());
            }
        });
        nsDictionaryMaker.saveTxtTo("D:\\JavaProjects\\HanLP\\data\\dictionary\\organization\\outerNT");
    }

    public void testSplitLargeFile() throws Exception
    {
        String line = null;
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/test/nt/company.txt")));
        int id = 1;
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/test/nt/part/" + id + ".txt")));
        int count = 1;
        while ((line = br.readLine()) != null)
        {
            if (count == 1000)
            {
                bw.close();
                bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/test/nt/part/" + id + ".txt")));
                ++id;
                count = 0;
            }
            bw.write(line);
            bw.newLine();
            ++count;
        }
        br.close();
    }

    public void testCase() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableOrganizationRecognize(true);
        System.out.println(segment.seg("黑龙江建筑职业技术学院近百学生发生冲突"));
    }

    public void testCombine() throws Exception
    {
        DictionaryMaker.combine("data/dictionary/organization/nt.txt", "data/dictionary/organization/outerNT.txt").saveTxtTo("data/dictionary/organization/nt.txt");
    }
}
