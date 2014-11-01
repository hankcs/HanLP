/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/3 12:06</create-date>
 *
 * <copyright file="TestCustomDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;

import java.io.*;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * @author hankcs
 */
public class TestCustomDictionary extends TestCase
{
    public void testGet() throws Exception
    {
        System.out.println(CustomDictionary.get("工信处"));
    }

    /**
     * 删除一个字的词语
     * @throws Exception
     */
    public void testRemoveShortWord() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/dictionary/CustomDictionary.txt")));
        String line;
        Set<String> fixedDictionary = new TreeSet<>();
        while ((line = br.readLine()) != null)
        {
            String[] param = line.split("\\s");
            if (param[0].length() == 1 || CoreDictionary.contains(param[0])) continue;
            fixedDictionary.add(line);
        }
        br.close();
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/dictionary/CustomDictionary.txt")));
        for (String word : fixedDictionary)
        {
            bw.write(word);
            bw.newLine();
        }
        bw.close();
    }

    /**
     * 这里面很多nr不合理，干脆都删掉
     * @throws Exception
     */
    public void testRemoveNR() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/dictionary/CustomDictionary.txt")));
        String line;
        Set<String> fixedDictionary = new TreeSet<>();
        while ((line = br.readLine()) != null)
        {
            String[] param = line.split("\\s");
            if (param[1].equals("nr")) continue;
            fixedDictionary.add(line);
        }
        br.close();
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/dictionary/CustomDictionary.txt")));
        for (String word : fixedDictionary)
        {
            bw.write(word);
            bw.newLine();
        }
        bw.close();
    }

    public void testNext() throws Exception
    {
        BaseSearcher searcher = CustomDictionary.getSearcher("都要亲口");
        Map.Entry<String, CoreDictionary.Attribute> entry;
        while ((entry = searcher.next()) != null)
        {
            int offset = searcher.getOffset();
            System.out.println(offset + 1 + " " + entry);
        }
    }
}
