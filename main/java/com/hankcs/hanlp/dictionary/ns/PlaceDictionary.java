/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 14:47</create-date>
 *
 * <copyright file="PersonDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.ns;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algoritm.ahocorasick.trie.Emit;
import com.hankcs.hanlp.algoritm.ahocorasick.trie.Trie;
import com.hankcs.hanlp.corpus.dictionary.item.EnumItem;
import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.corpus.tag.NS;
import com.hankcs.hanlp.dictionary.TransformMatrixDictionary;
import com.hankcs.hanlp.dictionary.nr.NRDictionary;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;

import java.util.Collection;
import java.util.List;
import java.util.ListIterator;

import static com.hankcs.hanlp.corpus.tag.NR.B;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 地名识别用的词典，实际上是对两个词典的包装
 *
 * @author hankcs
 */
public class PlaceDictionary
{
    /**
     * 地名词典
     */
    public static NSDictionary dictionary;
    /**
     * 转移矩阵词典
     */
    public static TransformMatrixDictionary<NS> transformMatrixDictionary;
    /**
     * AC算法用到的Trie树
     */
    public static Trie trie;

    static
    {
        long start = System.currentTimeMillis();
        dictionary = new NSDictionary();
        dictionary.load(HanLP.Config.PlaceDictionaryPath);
        logger.info(HanLP.Config.PlaceDictionaryPath + "加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
        transformMatrixDictionary = new TransformMatrixDictionary<NS>(NS.class);
        transformMatrixDictionary.load(HanLP.Config.PlaceDictionaryTrPath);
        trie = new Trie();
        trie.addKeyword("CH");
        trie.addKeyword("CDH");
        trie.addKeyword("CDEH");
        trie.addKeyword("GH");
    }

    /**
     * 模式匹配
     *
     * @param nsList         确定的标注序列
     * @param vertexList     原始的未加角色标注的序列
     * @param wordNetOptimum 待优化的图
     * @param wordNetAll
     */
    public static void parsePattern(List<NS> nsList, List<Vertex> vertexList, WordNet wordNetOptimum, WordNet wordNetAll)
    {
//        ListIterator<Vertex> listIterator = vertexList.listIterator();
        StringBuilder sbPattern = new StringBuilder(nsList.size());
        for (NS ns : nsList)
        {
            sbPattern.append(ns.toString());
        }
        String pattern = sbPattern.toString();
        Collection<Emit> emitCollection = trie.parseText(pattern);
        Vertex[] wordArray = vertexList.toArray(new Vertex[0]);
        for (Emit emit : emitCollection)
        {
            String keyword = emit.getKeyword();
            int start = emit.getStart();
            int end = emit.getEnd();
            StringBuilder sbName = new StringBuilder();
            for (int i = start; i <= end; ++i)
            {
                sbName.append(wordArray[i].realWord);
            }
            String name = sbName.toString();
            // 对一些bad case做出调整
//            if (isBadCase(name)) continue;

            // 正式算它是一个名字
            if (HanLP.Config.DEBUG)
            {
                System.out.printf("识别出地名：%s %s\n", name, keyword);
            }
            int offset = 0;
            for (int i = 0; i < start; ++i)
            {
                offset += wordArray[i].realWord.length();
            }
            wordNetOptimum.insert(offset, Vertex.newPlaceInstance(name, 1000), wordNetAll);
        }
    }

    /**
     * 因为任何算法都无法解决100%的问题，总是有一些bad case，这些bad case会以“盖公章 A 1”的形式加入词典中<BR>
     * 这个方法返回是否是bad case
     *
     * @param name
     * @return
     */
    static boolean isBadCase(String name)
    {
        EnumItem<NS> nrEnumItem = dictionary.get(name);
        if (nrEnumItem == null) return false;
        return nrEnumItem.containsLabel(NS.A);
    }
}
