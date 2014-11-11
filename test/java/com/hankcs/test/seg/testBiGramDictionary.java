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

import com.hankcs.hanlp.dictionary.BiGramDictionary;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class testBiGramDictionary extends TestCase
{
    public void testBiGramDictionary()
    {
//        assertEquals(15, BiGramDictionary.getBiFrequency("团结", "奋斗"));
//        assertEquals(1, BiGramDictionary.getBiFrequency("团结", "拼搏"));
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
        System.out.println(BiGramDictionary.getBiFrequency("亲@未##专"));
    }
}
