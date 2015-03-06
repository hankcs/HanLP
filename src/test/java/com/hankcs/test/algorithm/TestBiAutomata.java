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

import com.hankcs.hanlp.algoritm.automata.dat.BiAutomataDat;
import junit.framework.TestCase;

import java.util.TreeMap;
import java.util.TreeSet;

/**
 * @author hankcs
 */
public class TestBiAutomata extends TestCase
{
    public void testBuild() throws Exception
    {
        TreeMap<Integer, TreeSet<Integer>> map = new TreeMap<Integer, TreeSet<Integer>>();
        {
            TreeSet<Integer> value = new TreeSet<Integer>();
            value.add(3);
            value.add(5);
            map.put(1, value);
        }
        {
            TreeSet<Integer> value = new TreeSet<Integer>();
            value.add(4);
            map.put(2, value);
        }
        BiAutomataDat<Boolean> automata = new BiAutomataDat<Boolean>();
        automata.build(map);
        System.out.println(automata.transition(1, 3));
    }
}
