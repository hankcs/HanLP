/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/12 19:00</create-date>
 *
 * <copyright file="TestMakeJapaneseName.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.BiGramDictionary;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.nr.JapanesePersonDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.utility.TextUtility;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.util.*;

/**
 * @author hankcs
 */
public class TestMakeJapaneseName extends TestCase
{
    public void testCombine() throws Exception
    {
        String root = "D:\\JavaProjects\\SougouDownload\\data\\";
        String[] pathArray = new String[]{"日本名人大合集.txt", "日剧电影动漫和日本明星.txt", "日本女优.txt", "日本AV女优(A片)EXTEND版.txt", "日本女优大全.txt"};
        Set<String> wordSet = new TreeSet<String>();
        for (String path : pathArray)
        {
            path = root + path;
            for (String word : IOUtil.readLineList(path))
            {
                word = word.replaceAll("[a-z\r\n]", "");
                if (CoreDictionary.contains(word) || CustomDictionary.contains(word)) continue;
                wordSet.add(word);
            }
        }

        TreeSet<String> firstNameSet = new TreeSet<String>();
        firstNameSet.addAll(IOUtil.readLineList("data/dictionary/person/日本姓氏.txt"));
        Iterator<String> iterator = wordSet.iterator();
        while (iterator.hasNext())
        {
            String name = iterator.next();
            if (name.length() > 6 || name.length() < 3 || (!firstNameSet.contains(name.substring(0, 1)) && !firstNameSet.contains(name.substring(0, 2)) && !firstNameSet.contains(name.substring(0, 3))))
            {
                iterator.remove();
            }
        }

        IOUtil.saveCollectionToTxt(wordSet, "data/dictionary/person/日本人名.txt");
    }

    public void testMakeRoleDictionary() throws Exception
    {
        TreeSet<String> firstNameSet = new TreeSet<String>();
        firstNameSet.addAll(IOUtil.readLineList("data/dictionary/person/日本姓氏.txt"));
        TreeSet<String> fullNameSet = new TreeSet<String>();
        fullNameSet.addAll(IOUtil.readLineList("data/dictionary/person/日本人名.txt"));
        StringDictionary dictionary = new StringDictionary(" ");
        for (String fullName : fullNameSet)
        {
            for (int i = Math.min(3, fullName.length() - 1); i > 0; --i)
            {
                String firstName = fullName.substring(0, i);
                if (firstNameSet.contains(firstName))
                {
                    dictionary.add(fullName.substring(i), "m");
                    break;
                }
            }
        }
        for (String firstName : firstNameSet)
        {
            dictionary.add(firstName, "x");
        }
        dictionary.save("data/dictionary/person/nrj.txt");
    }

    public void testRecognize() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        System.out.println(segment.seg("我叫大杉亚依里"));
    }

    private String getLongestSuffix(String a, String b)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length() && i < b.length(); ++i)
        {
            if (a.charAt(i) == b.charAt(i))
            {
                sb.append(a.charAt(i));
            }
            else
            {
                return sb.toString();
            }
        }
        return sb.toString();
    }

    public void testImport() throws Exception
    {
        TreeSet<String> set = new TreeSet<String>();
        for (String name : IOUtil.readLineList("D:\\Doc\\语料库\\corpus-master\\日本姓氏.txt"))
        {
            name = HanLP.convertToSimplifiedChinese(Arrays.toString(name.toCharArray()));
            name = name.replaceAll("[\\[\\], ]", "");
            if (!TextUtility.isAllChinese(name)) continue;
            set.add(name);
        }
        IOUtil.saveCollectionToTxt(set, "data/dictionary/person/日本姓氏.txt");
    }

    public void testLoadJapanese() throws Exception
    {
        System.out.println(JapanesePersonDictionary.get("太郎"));
    }

    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableJapaneseNameRecognize(true);
        System.out.println(segment.seg("林志玲亮相网友:确定不是波多野结衣？"));
    }

    public void testCountBadCase() throws Exception
    {
        BufferedWriter bw = IOUtil.newBufferedWriter(HanLP.Config.JapanesePersonDictionaryPath + ".badcase.txt");
        List<String> xList = new LinkedList<String>();
        List<String> mList = new LinkedList<String>();
        IOUtil.LineIterator iterator = new IOUtil.LineIterator(HanLP.Config.JapanesePersonDictionaryPath);
        while (iterator.hasNext())
        {
            String line = iterator.next();
            String[] args = line.split("\\s");
            if ("x".equals(args[1])) xList.add(args[0]);
            else mList.add(args[0]);
        }
        for (String x : xList)
        {
            for (String m : mList)
            {
                if (CoreBiGramTableDictionary.getBiFrequency(x, m) > 0)
                {
                    bw.write(x + m + " A");
                    bw.newLine();
                }
            }
        }
        bw.close();
    }
}
