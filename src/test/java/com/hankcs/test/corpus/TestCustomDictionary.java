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

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

import java.io.*;
import java.util.*;

/**
 * @author hankcs
 */
public class TestCustomDictionary extends TestCase
{
    public void testGet() throws Exception
    {
        System.out.println(CustomDictionary.get("一个心眼儿"));
    }

    /**
     * 删除一个字的词语
     * @throws Exception
     */
    public void testRemoveShortWord() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/dictionary/CustomDictionary.txt")));
        String line;
        Set<String> fixedDictionary = new TreeSet<String>();
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
        Set<String> fixedDictionary = new TreeSet<String>();
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

    public void testRemoveJunkWord() throws Exception
    {
        DictionaryMaker dictionaryMaker = DictionaryMaker.load("data/dictionary/custom/CustomDictionary.txt");
        dictionaryMaker.saveTxtTo("data/dictionary/custom/CustomDictionary.txt", new DictionaryMaker.Filter()
        {
            @Override
            public boolean onSave(Item item)
            {
                if (item.containsLabel("mq") || item.containsLabel("m") || item.containsLabel("t"))
                {
                    return false;
                }
                return true;
            }
        });
    }

    /**
     * data/dictionary/custom/全国地名大全.txt中有很多人名，删掉它们
     * @throws Exception
     */
    public void testRemoveNotNS() throws Exception
    {
        String path = "data/dictionary/custom/全国地名大全.txt";
        final Set<Character> suffixSet = new TreeSet<Character>();
        for (char c : Predefine.POSTFIX_SINGLE.toCharArray())
        {
            suffixSet.add(c);
        }
        DictionaryMaker.load(path).saveTxtTo(path, new DictionaryMaker.Filter()
        {
            Segment segment = HanLP.newSegment().enableCustomDictionary(false);
            @Override
            public boolean onSave(Item item)
            {
                if (suffixSet.contains(item.key.charAt(item.key.length() - 1))) return true;
                List<Term> termList = segment.seg(item.key);
                if (termList.size() == 1 && termList.get(0).nature == Nature.nr)
                {
                    System.out.println(item);
                    return false;
                }
                return true;
            }
        });
    }

    public void testCustomNature() throws Exception
    {
        Nature pcNature1 = Nature.create("电脑品牌");
        Nature pcNature2 = Nature.create("电脑品牌");
        assertEquals(pcNature1, pcNature2);
    }

    public void testIssue234() throws Exception
    {
        String customTerm = "攻城狮";
        String text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰";
        System.out.println("原始分词结果");
        System.out.println("CustomDictionary.get(customTerm)=" + CustomDictionary.get(customTerm));
        System.out.println(HanLP.segment(text));
        // 动态增加
        CustomDictionary.add(customTerm);
        System.out.println("添加自定义词组分词结果");
        System.out.println("CustomDictionary.get(customTerm)=" + CustomDictionary.get(customTerm));
        System.out.println(HanLP.segment(text));
        // 删除词语
        CustomDictionary.remove(customTerm);
        System.out.println("删除自定义词组分词结果");
        System.out.println("CustomDictionary.get(customTerm)=" + CustomDictionary.get(customTerm));
        System.out.println(HanLP.segment(text));
    }

    public void testIssue540() throws Exception
    {
        CustomDictionary.add("123");
        CustomDictionary.add("摩根");
        CustomDictionary.remove("123");
        CustomDictionary.remove("摩根");
    }
}
