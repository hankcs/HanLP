/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 23:51</create-date>
 *
 * <copyright file="TestMakePinYinDictioanry.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.SimpleDictionary;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.dictionary.StringDictionaryMaker;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.py.*;
import com.hankcs.hanlp.utility.TextUtility;
import junit.framework.TestCase;

import java.util.*;

/**
 * @author hankcs
 */
public class TestMakePinYinDictionary extends TestCase
{
//    public void testCombine() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        StringDictionary dictionaryPY = new StringDictionary();
//        dictionaryPY.load("D:\\JavaProjects\\jpinyin\\data\\pinyinTable.standard.txt");
//
////        StringDictionary dictionaryAnsj = new StringDictionary();
////        dictionaryAnsj.load("D:\\JavaProjects\\jpinyin\\data\\ansj.txt");
////        System.out.println(dictionaryAnsj.remove(new SimpleDictionary.Filter()
////        {
////            @Override
////            public boolean remove(Map.Entry entry)
////            {
////                return entry.getValue().toString().endsWith("0");
////            }
////        }));
//
//        StringDictionary dictionaryPolyphone = new StringDictionary();
//        dictionaryPolyphone.load("D:\\JavaProjects\\jpinyin\\data\\polyphone.txt");
//
//        StringDictionary dictionarySingle = new StringDictionary();
//        dictionarySingle.load("data/dictionary/pinyin/single.txt");
//
//        StringDictionary main = StringDictionaryMaker.combine(dictionaryPY, dictionaryPolyphone, dictionarySingle);
//        main.save("data/dictionary/pinyin/pinyin.txt");
//    }
//
//    public void testCombineSingle() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        StringDictionary main = StringDictionaryMaker.combine("data/dictionary/pinyin/pinyin.txt", "data/dictionary/pinyin/single.txt");
//        main.save("data/dictionary/pinyin/pinyin.txt");
//    }
//
//    public void testSpeed() throws Exception
//    {
//
//    }
//
//
//    public void testMakeSingle() throws Exception
//    {
//        LinkedList<String[]> csv = IOUtil.readCsv("D:\\JavaProjects\\jpinyin\\data\\words.csv");
//        StringDictionary dictionarySingle = new StringDictionary();
//        for (String[] args : csv)
//        {
//            //  0    1  2     3  4  5   6      7
//            // 6895,中,zhong,zh,ong,1,\u4E2D,中 zhong \u4E2D
//            String word = args[1];
//            String py = args[2];
//            String sm = args[3];
//            String ym = args[4];
//            String yd = args[5];
//            String pyyd = py + yd;
//            // 过滤
//            if (!TextUtility.isAllChinese(word)) continue;
//            dictionarySingle.add(word, pyyd);
//        }
//        dictionarySingle.save("data/dictionary/pinyin/single.txt");
//    }
//
//    public void testMakeTable() throws Exception
//    {
//        LinkedList<String[]> csv = IOUtil.readCsv("D:\\JavaProjects\\jpinyin\\data\\words.csv");
//        StringDictionary dictionarySingle = new StringDictionary();
//        for (String[] args : csv)
//        {
//            //  0    1  2     3  4  5   6      7
//            // 6895,中,zhong,zh,ong,1,\u4E2D,中 zhong \u4E2D
//            String word = args[1];
//            String py = args[2];
//            String sm = args[3];
//            String ym = args[4];
//            String yd = args[5];
//            String pyyd = py + yd;
//            // 过滤
//            if (!TextUtility.isAllChinese(word)) continue;
//            dictionarySingle.add(pyyd, sm + "," + ym + "," + yd);
//        }
//        dictionarySingle.save("data/dictionary/pinyin/sm-ym-table.txt");
//    }
//
//    public void testConvert() throws Exception
//    {
//        String text = "重载不是重担，" + HanLP.convertToTraditionalChinese("以后爱皇后");
//        List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(text);
//        System.out.print("原文,");
//        for (char c : text.toCharArray())
//        {
//            System.out.printf("%c,", c);
//        }
//        System.out.println();
//
//        System.out.print("拼音（数字音调）,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin);
//        }
//        System.out.println();
//
//        System.out.print("拼音（符号音调）,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getPinyinWithToneMark());
//        }
//        System.out.println();
//
//        System.out.print("拼音（无音调）,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getPinyinWithoutTone());
//        }
//        System.out.println();
//
//        System.out.print("声调,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getTone());
//        }
//        System.out.println();
//
//        System.out.print("声母,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getShengmu());
//        }
//        System.out.println();
//
//        System.out.print("韵母,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getYunmu());
//        }
//        System.out.println();
//
//        System.out.print("输入法头,");
//        for (Pinyin pinyin : pinyinList)
//        {
//            System.out.printf("%s,", pinyin.getHeadString());
//        }
//        System.out.println();
//    }
//
//    public void testMakePinyinEnum() throws Exception
//    {
//        StringDictionary dictionary = new StringDictionary();
//        dictionary.load("data/dictionary/pinyin/pinyin.txt");
//
//        StringDictionary pyEnumDictionary = new StringDictionary();
//        for (Map.Entry<String, String> entry : dictionary.entrySet())
//        {
//            String[] args = entry.getValue().split(",");
//            for (String arg : args)
//            {
//                pyEnumDictionary.add(arg, arg);
//            }
//        }
//
//        StringDictionary table = new StringDictionary();
//        table.combine(pyEnumDictionary);
//
//        StringBuilder sb = new StringBuilder();
//        for (Map.Entry<String, String> entry : table.entrySet())
//        {
//            sb.append(entry.getKey());
//            sb.append('\n');
//        }
//        IOUtil.saveTxt("data/dictionary/pinyin/py.enum.txt", sb.toString());
//    }
//
//    /**
//     * 有些拼音没有声母和韵母，尝试根据上文拓展它们
//     * @throws Exception
//     */
//    public void testExtendTable() throws Exception
//    {
//        StringDictionary dictionary = new StringDictionary();
//        dictionary.load("data/dictionary/pinyin/pinyin.txt");
//
//        StringDictionary pyEnumDictionary = new StringDictionary();
//        for (Map.Entry<String, String> entry : dictionary.entrySet())
//        {
//            String[] args = entry.getValue().split(",");
//            for (String arg : args)
//            {
//                pyEnumDictionary.add(arg, arg);
//            }
//        }
//
//        StringDictionary table = new StringDictionary();
//        table.load("data/dictionary/pinyin/sm-ym-table.txt");
//        table.combine(pyEnumDictionary);
//
//        Iterator<Map.Entry<String, String>> iterator = table.entrySet().iterator();
//        Map.Entry<String, String> pre = iterator.next();
//        String prePy = pre.getKey().substring(0, pre.getKey().length() - 1);
//        String preYd = pre.getKey().substring(pre.getKey().length() - 1);
//        while (iterator.hasNext())
//        {
//            Map.Entry<String, String> current = iterator.next();
//            String currentPy = current.getKey().substring(0, current.getKey().length() - 1);
//            String currentYd = current.getKey().substring(current.getKey().length() - 1);
//            // handle it
//            if (!current.getValue().contains(","))
//            {
//                if (currentPy.equals(prePy))
//                {
//                    table.add(current.getKey(), pre.getValue().replace(preYd, currentYd));
//                }
//                else
//                {
//                    System.out.println(currentPy + currentYd);
//                }
//            }
//            // end
//            pre = current;
//            prePy = currentPy;
//            preYd = currentYd;
//        }
//        table.save("data/dictionary/pinyin/sm-ym-yd-table.txt");
//    }
//
//    public void testDumpSMT() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        SYTDictionary.dumpEnum("data/dictionary/pinyin/");
//    }
//
//    public void testPinyinDictionary() throws Exception
//    {
//        HanLP.Config.enableDebug();
//        Pinyin[] pinyins = PinyinDictionary.get("中");
//        System.out.println(Arrays.toString(pinyins));
//    }
//
//    public void testCombineAnsjWithPinyinTxt() throws Exception
//    {
//        StringDictionary dictionaryAnsj = new StringDictionary();
//        dictionaryAnsj.load("D:\\JavaProjects\\jpinyin\\data\\ansj.txt");
//        System.out.println(dictionaryAnsj.remove(new SimpleDictionary.Filter<String>()
//        {
//            @Override
//            public boolean remove(Map.Entry<String, String> entry)
//            {
//                String word = entry.getKey();
//                String pinyin = entry.getValue();
//                String[] pinyinStringArray = entry.getValue().split("[,\\s　]");
//                if (word.length() != pinyinStringArray.length || !TonePinyinString2PinyinConverter.valid(pinyinStringArray))
//                {
//                    System.out.println(entry);
//                    return false;
//                }
//
//                return true;
//            }
//        }));
//
//    }
//
//    public void testMakePinyinJavaCode() throws Exception
//    {
//        StringBuilder sb = new StringBuilder();
//        for (Pinyin pinyin : PinyinDictionary.pinyins)
//        {
//            // 0声母 1韵母 2音调 3带音标
//            sb.append(pinyin + "(" + Shengmu.class.getSimpleName() + "." + pinyin.getShengmu() + ", " + Yunmu.class.getSimpleName() + "." + pinyin.getYunmu() + ", " + pinyin.getTone() + ", \"" + pinyin.getPinyinWithToneMark() + "\", \"" + pinyin.getPinyinWithoutTone() + "\"" + ", " + Head.class.getSimpleName() + "." + pinyin.getHeadString() + ", '" + pinyin.getFirstChar() + "'" + "),\n");
//        }
//        IOUtil.saveTxt("data/dictionary/pinyin/py.txt", sb.toString());
//    }
//
//    public void testConvertUnicodeTable() throws Exception
//    {
//        StringDictionary dictionary = new StringDictionary("=");
//        for (String line : IOUtil.readLineList("D:\\Doc\\语料库\\Uni2Pinyin.txt"))
//        {
//            if (line.startsWith("#")) continue;
//            String[] argArray = line.split("\\s");
//            if (argArray.length == 1) continue;
//            String py = argArray[1];
//            for (int i = 2; i < argArray.length; ++i)
//            {
//                py += ',';
//                py += argArray[i];
//            }
//            dictionary.add(String.valueOf((char)(Integer.parseInt(argArray[0], 16))), py);
//        }
//        dictionary.save("D:\\Doc\\语料库\\Hanzi2Pinyin.txt");
//    }
//
//    public void testCombineUnicodeTableWithMainDictionary() throws Exception
//    {
//        StringDictionary mainDictionary = new StringDictionary("=");
//        mainDictionary.load("data/dictionary/pinyin/pinyin.txt");
//        StringDictionary subDictionary = new StringDictionary("=");
//        subDictionary.load("D:\\Doc\\语料库\\Hanzi2Pinyin.txt");
//        mainDictionary.combine(subDictionary);
//        mainDictionary.save("data/dictionary/pinyin/pinyin.txt");
//    }
}
