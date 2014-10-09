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
package com.hankcs.test.corpus;

import junit.framework.TestCase;

import java.io.*;
import java.util.*;

/**
 * 有一些类似于 工程@学 1 的条目会干扰 工程学家 的识别，这类@后接短字符的可以过滤掉
 * @author hankcs
 */
public class SimplifyNGramDictionary extends TestCase
{
    String path = "data/dictionary/CoreNatureDictionary.ngram.txt";
    public void testSimplify() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        TreeMap<String, Integer> map = new TreeMap<>();
        String line;
        while ((line = br.readLine()) != null)
        {
            String[] param = line.split("\\s");
            map.put(param[0], Integer.valueOf(param[1]));
        }
        br.close();
        Set<Map.Entry<String, Integer>> entrySet = map.descendingMap().entrySet();
        // 第一步去包含
        Iterator<Map.Entry<String, Integer>> iterator = entrySet.iterator();
        Map.Entry<String, Integer> pre = new AbstractMap.SimpleEntry<>(" @ ", 1);
        while (iterator.hasNext())
        {
            Map.Entry<String, Integer> current = iterator.next();
            if (current.getKey().length() - current.getKey().indexOf('@') == 2 && pre.getKey().indexOf(current.getKey()) == 0 && current.getValue() <= 2)
            {
                System.out.println("应当删除 " + current + " 保留 " + pre);
                iterator.remove();
            }
            pre = current;
        }
        // 第二步，尝试移除“学@家”这样的短共现
//        iterator = entrySet.iterator();
//        while (iterator.hasNext())
//        {
//            Map.Entry<String, Integer> current = iterator.next();
//            if (current.getKey().length() == 3)
//            {
//                System.out.println("应当删除 " + current);
//            }
//        }
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
        for (Map.Entry<String, Integer> entry : map.entrySet())
        {
            bw.write(entry.getKey());
            bw.write(' ');
            bw.write(String.valueOf(entry.getValue()));
            bw.newLine();
        }
        bw.close();
    }
}
