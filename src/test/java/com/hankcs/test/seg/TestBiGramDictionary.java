/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/16 21:40</create-date>
 *
 * <copyright file="testBiGramDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.dictionary.BiGramDictionary;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import junit.framework.TestCase;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * @author hankcs
 */
public class TestBiGramDictionary extends TestCase
{
    public void testBiGramDictionary()
    {
        assertEquals(15, BiGramDictionary.getBiFrequency("团结", "奋斗"));
        assertEquals(1, BiGramDictionary.getBiFrequency("团结", "拼搏"));
//        BufferedReader br = null;
//        try
//        {
//            br = new BufferedReader(new InputStreamReader(new FileInputStream(BiGramDictionary.path)));
//            String line;
//            while ((line = br.readLine()) != null)
//            {
//                String[] params = line.split("\t");
//                String twoWord = params[0];
//                int freq = Integer.parseInt(params[1]);
//                assertEquals(freq, BiGramDictionary.getBiFrequency(twoWord));
//            }
//            br.close();
//        } catch (FileNotFoundException e)
//        {
////            LogManager.getLogger().fatal("二元词典不存在！");
//            e.printStackTrace();
//        } catch (IOException e)
//        {
////            LogManager.getLogger().fatal("二元词典读取错误！");
//            e.printStackTrace();
//        }
//
//        // 测试不存在的键
//        assertEquals(0, BiGramDictionary.getBiFrequency("不存在"));
        HanLP.Config.enableDebug();
        BiGramDictionary.getBiFrequency("团结@奋斗");
    }

    public void testTable() throws Exception
    {
        assertEquals(106, CoreBiGramTableDictionary.getBiFrequency("延续", "未##时"));
    }

    /**
     * 测试两者兼容性，顺便将二元词典中多出来的词语记录下来，可以回写到核心词典中
     * @throws Exception
     */
    public void testFastBiGram() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(HanLP.Config.BiGramDictionaryPath)));
        String line;
        DictionaryMaker dictionaryMaker = new DictionaryMaker();
        double total = 0;
        double right = 0;
        while ((line = br.readLine()) != null)
        {
            ++total;
            String[] params = line.split("\\s");
            String[] twoWord = params[0].split("@", 2);
            String a = twoWord[0];
            String b = twoWord[1];
            int idA = CoreBiGramTableDictionary.getWordID(a);
            int idB = CoreBiGramTableDictionary.getWordID(b);
//            assert BiGramDictionary.getBiFrequency(a, b) == CoreBiGramDictionary.getBiFrequency(a, b) : line;
            if (BiGramDictionary.getBiFrequency(a, b) != CoreBiGramTableDictionary.getBiFrequency(idA, idB))
            {
                System.out.println(line);
                if (idA < 0)
                {
                    dictionaryMaker.add(a, "n");
                }
                if (idB < 0)
                {
                    dictionaryMaker.add(b, "n");
                }
            }
            else
            {
                ++right;
            }
        }
        br.close();
        System.out.println("覆盖率：" + (right / total));
        dictionaryMaker.saveTxtTo("data/test/out.txt");
    }

    public void testSingle() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(CoreBiGramTableDictionary.getBiFrequency("团结", "奋斗"));
    }

    public void testBenchmark() throws Exception
    {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(HanLP.Config.BiGramDictionaryPath)));
        String line;
        List<String[]> twoWordList = new LinkedList<String[]>();
        while ((line = br.readLine()) != null)
        {
            String[] params = line.split("\\s");
            String[] twoWord = params[0].split("@", 2);
            twoWordList.add(twoWord);
        }
        br.close();
        long start = System.currentTimeMillis();
        for (String[] twoWord : twoWordList)
        {
        }
    }

    public void testObjectOutPut() throws Exception
    {
        int size = 5563418;
        int[] array = new int[size];
        for (int i = 0; i < array.length; i++)
        {
            array[i] = i;
        }
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("data/test/out.bin"));
        out.writeObject(array);
        out.close();

        long start = System.currentTimeMillis();
        ObjectInputStream in = new ObjectInputStream(new FileInputStream("data/test/out.bin"));
        int[] inArray = (int[]) in.readObject();
        System.out.println(System.currentTimeMillis() - start);
        for (int i = 0; i < inArray.length; i++)
        {
            assertEquals(i, inArray[i]);
        }
    }
}
