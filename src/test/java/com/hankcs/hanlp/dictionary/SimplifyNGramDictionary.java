/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/7 21:06</create-date>
 *
 * <copyright file="SimplifyNGramDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.TFDictionary;
import com.hankcs.hanlp.corpus.occurrence.TermFrequency;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import junit.framework.TestCase;

import java.io.*;
import java.util.*;

/**
 * 有一些类似于 工程@学 1 的条目会干扰 工程学家 的识别，这类@后接短字符的可以过滤掉
 * @author hankcs
 */
public class SimplifyNGramDictionary extends TestCase
{
//    String path = "data/dictionary/CoreNatureDictionary.ngram.txt";
//    public void testSimplify() throws Exception
//    {
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
//        TreeMap<String, Integer> map = new TreeMap<String, Integer>();
//        String line;
//        while ((line = br.readLine()) != null)
//        {
//            String[] param = line.split("\\s");
//            map.put(param[0], Integer.valueOf(param[1]));
//        }
//        br.close();
//        Set<Map.Entry<String, Integer>> entrySet = map.descendingMap().entrySet();
//        Iterator<Map.Entry<String, Integer>> iterator = entrySet.iterator();
//        // 第一步去包含
////        Map.Entry<String, Integer> pre = new AbstractMap.SimpleEntry<>(" @ ", 1);
////        while (iterator.hasNext())
////        {
////            Map.Entry<String, Integer> current = iterator.next();
////            if (current.getKey().length() - current.getKey().indexOf('@') == 2 && pre.getKey().indexOf(current.getKey()) == 0 && current.getValue() <= 2)
////            {
////                System.out.println("应当删除 " + current + " 保留 " + pre);
////                iterator.remove();
////            }
////            pre = current;
////        }
//        // 第二步，尝试移除“学@家”这样的短共现
////        iterator = entrySet.iterator();
////        while (iterator.hasNext())
////        {
////            Map.Entry<String, Integer> current = iterator.next();
////            if (current.getKey().length() == 3)
////            {
////                System.out.println("应当删除 " + current);
////            }
////        }
//        // 第三步，对某些@后面的词语太短了，也移除
////        iterator = entrySet.iterator();
////        while (iterator.hasNext())
////        {
////            Map.Entry<String, Integer> current = iterator.next();
////            String[] termArray = current.getKey().split("@", 2);
////            if (termArray[0].equals("未##人") && termArray[1].length() < 2)
////            {
////                System.out.println("删除 " + current.getKey());
////                iterator.remove();
////            }
////        }
//        // 第四步，人名接续对识别产生太多误命中影响，也删除
////        iterator = entrySet.iterator();
////        while (iterator.hasNext())
////        {
////            Map.Entry<String, Integer> current = iterator.next();
////            if (current.getKey().contains("未##人") && current.getValue() < 10)
////            {
////                System.out.println("删除 " + current.getKey());
////                iterator.remove();
////            }
////        }
//        // 对人名的终极调优
//        TFDictionary dictionary = new TFDictionary();
//        dictionary.load("D:\\JavaProjects\\HanLP\\data\\dictionary\\CoreNatureDictionary.ngram.mini.txt");
//        iterator = entrySet.iterator();
//        while (iterator.hasNext())
//        {
//            Map.Entry<String, Integer> current = iterator.next();
//            if (current.getKey().contains("未##人") && dictionary.getFrequency(current.getKey()) < 10)
//            {
//                System.out.println("删除 " + current.getKey());
//                iterator.remove();
//            }
//        }
//        // 输出
//        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
//        for (Map.Entry<String, Integer> entry : map.entrySet())
//        {
//            bw.write(entry.getKey());
//            bw.write(' ');
//            bw.write(String.valueOf(entry.getValue()));
//            bw.newLine();
//        }
//        bw.close();
//    }
//
//    /**
//     * 有些词条不在CoreDictionary里面，那就把它们删掉
//     * @throws Exception
//     */
//    public void testLoseWeight() throws Exception
//    {
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
//        TreeMap<String, Integer> map = new TreeMap<String, Integer>();
//        String line;
//        while ((line = br.readLine()) != null)
//        {
//            String[] param = line.split(" ");
//            map.put(param[0], Integer.valueOf(param[1]));
//        }
//        br.close();
//        Iterator<String> iterator = map.keySet().iterator();
//        while (iterator.hasNext())
//        {
//            line = iterator.next();
//            String[] params = line.split("@", 2);
//            String one = params[0];
//            String two = params[1];
//            if (!CoreDictionary.contains(one) || !CoreDictionary.contains(two))
//                iterator.remove();
//        }
//
//        // 输出
//        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));
//        for (Map.Entry<String, Integer> entry : map.entrySet())
//        {
//            bw.write(entry.getKey());
//            bw.write(' ');
//            bw.write(String.valueOf(entry.getValue()));
//            bw.newLine();
//        }
//        bw.close();
//    }
}
