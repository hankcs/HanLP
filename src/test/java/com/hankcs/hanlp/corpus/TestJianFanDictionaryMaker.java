/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 19:46</create-date>
 *
 * <copyright file="TestJianFanDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.other.CharTable;
import junit.framework.TestCase;

import java.io.*;
import java.util.*;

/**
 * @author hankcs
 */
public class TestJianFanDictionaryMaker extends TestCase
{

    private String cc = "/Users/hankcs/CppProjects/OpenCC/data/dictionary/";
    private String root = "data/dictionary/tc/";

    public void testCombine() throws Exception
    {
//        StringDictionary dictionaryHanLP = new StringDictionary("=");
//        dictionaryHanLP.load(HanLP.Config.t2sDictionaryPath);
//
//        StringDictionary dictionaryOuter = new StringDictionary("=");
//        dictionaryOuter.load("D:\\Doc\\语料库\\简繁分歧词表.txt");
//
//        for (Map.Entry<String, String> entry : dictionaryOuter.entrySet())
//        {
//            String t = entry.getKey();
//            String s = entry.getValue();
//            if (t.length() == 1) continue;
//            if (HanLP.convertToTraditionalChinese(s).equals(t)) continue;
//            dictionaryHanLP.add(t, s);
//        }
//
//        dictionaryHanLP.save(HanLP.Config.t2sDictionaryPath);
    }

//    public void testConvertSingle() throws Exception
//    {
//        System.out.println(HanLP.convertToTraditionalChinese("一个劲"));
//    }
//
//    public void testIssue() throws Exception
//    {
//        System.out.println(HanLP.convertToSimplifiedChinese("缐"));
//        System.out.println(CharTable.convert("缐"));
//    }
//
//    public void testImportOpenCC() throws Exception
//    {
//        // 转换OpenCC的词库
//        Map<String, String> s2t = new TreeMap<String, String>();
//        combine("\t", s2t, cc + "STCharacters.txt",
//                cc + "STPhrases.txt"
//        );
//        save(s2t, "data/dictionary/tc/s2t.txt");
//        Map<String, String> t2s = new TreeMap<String, String>();
//        combine("=", t2s, "data/dictionary/tc/TraditionalChinese.txt");
//        combine("\t", t2s, cc + "TSCharacters.txt",
//                cc + "TSPhrases.txt"
//        );
//        save(t2s, "data/dictionary/tc/t2s.txt");
//    }
//
//    public void testMakeHK() throws Exception
//    {
//        Map<String, String> t2hk = new TreeMap<String, String>();
//        combine("\t", t2hk,
//                cc + "HKVariantsPhrases.txt",
//                cc + "HKVariants.txt"
//                );
//        save(t2hk, "data/dictionary/tc/t2hk.txt");
//    }
//
//    public void testMakeTW() throws Exception
//    {
//        Map<String, String> t2tw = new TreeMap<String, String>();
//        combine("\t", t2tw,
//                cc + "TWPhrasesIT.txt",
//                cc + "TWPhrasesName.txt",
//                cc + "TWPhrasesOther.txt",
//                cc + "TWVariants.txt"
//        );
//        save(t2tw, "data/dictionary/tc/t2tw.txt");
//    }
//
//    private void save(Map<String, String> storage, String path) throws IOException
//    {
//        BufferedWriter bw = IOUtil.newBufferedWriter(path);
//        for (Map.Entry<String, String> entry : storage.entrySet())
//        {
//            String line = entry.toString();
//            int firstBlank = line.indexOf(' ');
//            if (firstBlank != -1)
//            {
//                line = line.substring(0, firstBlank);
//            }
//            bw.write(line);
//            bw.newLine();
//        }
//        bw.close();
//    }
//
//    private Map<String, Set<String>> combine(Map<String, String> s2t, Map<String, String> t2s)
//    {
//        Map<String, Set<String>> all = new TreeMap<String, Set<String>>();
//        for (Map.Entry<String, String> entry : s2t.entrySet())
//        {
//            String key = entry.getKey();
//            Set<String> value = all.get(key);
//            if (value == null)
//            {
//                value = new TreeSet<String>();
//                all.put(key, value);
//            }
//            for (String v : entry.getValue().split(" "))
//            {
//                if (key.length() == 1 && key.equals(v))
//                {
//                    continue;
//                }
//                value.add(v);
//            }
//        }
//
//        for (Map.Entry<String, String> entry : t2s.entrySet())
//        {
//            for (String key : entry.getValue().split(" "))
//            {
//                if (key.length() == 1 && key.equals(entry.getKey()))
//                {
//                    continue;
//                }
//                Set<String> value = all.get(key);
//                if (value == null)
//                {
//                    value = new TreeSet<String>();
//                    all.put(key, value);
//                }
//
//                value.add(entry.getKey());
//            }
//        }
//
//        return all;
//    }
//
//    private Map<String, String> combine(String delimiter, Map<String, String> storage, String... pathArray)
//    {
//        for (String path : pathArray)
//        {
//            IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(path);
//            while (lineIterator.hasNext())
//            {
//                String line = lineIterator.next();
//                String[] args = line.split(delimiter);
//                if (args.length != 2)
//                {
//                    System.err.println(line);
//                    System.exit(-1);
//                }
//                storage.put(args[0], args[1]);
//            }
//        }
//
//        return storage;
//    }
//
//    public void testChar() throws Exception
//    {
//        String line = "㐹\t㑶 㐹";
//        System.out.println('㐹' == '㐹');
//    }
}
