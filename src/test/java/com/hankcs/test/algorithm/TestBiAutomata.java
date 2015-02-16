/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/2/16 22:54</create-date>
 *
 * <copyright file="TestBiAutomata.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.algorithm;

import com.hankcs.hanlp.algoritm.automata.BiAutomata;
import junit.framework.TestCase;

import java.util.TreeMap;

/**
 * @author hankcs
 */
public class TestBiAutomata extends TestCase
{
    public void testBuild() throws Exception
    {
        TreeMap<Integer, TreeMap<Integer, Integer>> map = new TreeMap<Integer, TreeMap<Integer, Integer>>();
        {
            TreeMap<Integer, Integer> value = new TreeMap<Integer, Integer>();
            value.put(3, 3);
            value.put(5, 5);
            map.put(1, value);
        }
        {
            TreeMap<Integer, Integer> value = new TreeMap<Integer, Integer>();
            value.put(4, 4);
            map.put(2, value);
        }
        BiAutomata<Integer> automata = new BiAutomata<Integer>();
        automata.build(map);
        System.out.println(automata.get(2, 4));
    }
}
