/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 17:53</create-date>
 *
 * <copyright file="Node.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.common;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.dependency.CoNll.PosTagCompiler;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;

import java.util.Map;
import java.util.TreeMap;

/**
 * 节点
 * @author hankcs
 */
public class Node
{
    private final static Map<String, String> natureConverter = new TreeMap<String, String>();
    static
    {
        natureConverter.put("begin", "root");
        natureConverter.put("bg", "b");
        natureConverter.put("e", "y");
        natureConverter.put("g", "nz");
        natureConverter.put("gb", "nz");
        natureConverter.put("gbc", "nz");
        natureConverter.put("gc", "nz");
        natureConverter.put("gg", "nz");
        natureConverter.put("gi", "nz");
        natureConverter.put("gm", "nz");
        natureConverter.put("gp", "nz");
        natureConverter.put("i", "nz");
        natureConverter.put("j", "nz");
        natureConverter.put("l", "nz");
        natureConverter.put("mg", "Mg");
        natureConverter.put("nb", "nz");
        natureConverter.put("nba", "nz");
        natureConverter.put("nbc", "nz");
        natureConverter.put("nbp", "nz");
        natureConverter.put("nf", "n");
        natureConverter.put("nh", "nz");
        natureConverter.put("nhd", "nz");
        natureConverter.put("nhm", "nz");
        natureConverter.put("ni", "nt");
        natureConverter.put("nic", "nt");
        natureConverter.put("nis", "n");
        natureConverter.put("nit", "nt");
        natureConverter.put("nm", "n");
        natureConverter.put("nmc", "nz");
        natureConverter.put("nn", "n");
        natureConverter.put("nnd", "n");
        natureConverter.put("nnt", "n");
        natureConverter.put("ntc", "nt");
        natureConverter.put("ntcb", "nt");
        natureConverter.put("ntcf", "nt");
        natureConverter.put("ntch", "nt");
        natureConverter.put("nth", "nt");
        natureConverter.put("nto", "nt");
        natureConverter.put("nts", "nt");
        natureConverter.put("ntu", "nt");
        natureConverter.put("nx", "x");
        natureConverter.put("qg", "q");
        natureConverter.put("rg", "Rg");
        natureConverter.put("ud", "u");
        natureConverter.put("udh", "u");
        natureConverter.put("ug", "uguo");
        natureConverter.put("uj", "u");
        natureConverter.put("ul", "ulian");
        natureConverter.put("uv", "u");
        natureConverter.put("uz", "uzhe");
        natureConverter.put("w", "x");
        natureConverter.put("wb", "x");
        natureConverter.put("wd", "x");
        natureConverter.put("wf", "x");
        natureConverter.put("wh", "x");
        natureConverter.put("wj", "x");
        natureConverter.put("wky", "x");
        natureConverter.put("wkz", "x");
        natureConverter.put("wm", "x");
        natureConverter.put("wn", "x");
        natureConverter.put("wp", "x");
        natureConverter.put("ws", "x");
        natureConverter.put("wt", "x");
        natureConverter.put("ww", "x");
        natureConverter.put("wyy", "x");
        natureConverter.put("wyz", "x");
        natureConverter.put("xu", "x");
        natureConverter.put("xx", "x");
        natureConverter.put("yg", "y");
        natureConverter.put("zg", "z");
    }
    public final static Node NULL = new Node(new Term(CoNLLWord.NULL.NAME, Nature.n), -1);
    static
    {
        NULL.label = "null";
    }
    public String word;
    public String compiledWord;
    public String label;
    public int id;

    public Node(Term term, int id)
    {
        this.id = id;
        word = term.word;
        label = natureConverter.get(term.nature.toString());
        if (label == null)
            label = term.nature.toString();
        compiledWord = PosTagCompiler.compile(label, word);
    }

    @Override
    public String toString()
    {
        return word + "/" + label;
    }
}
