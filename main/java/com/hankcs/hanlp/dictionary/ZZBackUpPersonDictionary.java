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
package com.hankcs.hanlp.dictionary;

import com.hankcs.hanlp.corpus.tag.NR;
import com.hankcs.hanlp.seg.NShort.Path.Vertex;
import com.hankcs.hanlp.seg.NShort.Path.WordNet;
import org.ahocorasick.trie.Emit;
import org.ahocorasick.trie.Trie;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;
import java.util.ListIterator;

import static com.hankcs.hanlp.corpus.tag.NR.B;
import static com.hankcs.hanlp.corpus.tag.NR.L;

/**
 * 人名识别用的词典，实际上是对两个词典的包装
 * @author hankcs
 */
public class ZZBackUpPersonDictionary
{
    static Logger logger = LoggerFactory.getLogger(ZZBackUpPersonDictionary.class);
    /**
     * 人名词典
     */
    public static NRDictionary dictionary;
    /**
     * 转移矩阵词典
     */
    public static TransformMatrixDictionary<NR> transformMatrixDictionary;
    /**
     * AC算法用到的Trie树
     */
    public static Trie trie;

    static
    {
        dictionary = new NRDictionary();
        dictionary.load("data/dictionary/person/combined.txt");
        transformMatrixDictionary = new TransformMatrixDictionary<>(NR.class);
        transformMatrixDictionary.load("data/dictionary/person/nr.tr.txt");
        trie = new Trie().removeOverlaps();
        trie.addKeyword("BBCD");
        trie.addKeyword("BBE");
        trie.addKeyword("BBZ");
        trie.addKeyword("BCD");
        trie.addKeyword("BEE");
        trie.addKeyword("BE");
        trie.addKeyword("BG");
        trie.addKeyword("BXD");
        trie.addKeyword("BZ");
        trie.addKeyword("CD");
        trie.addKeyword("EE");
        trie.addKeyword("FB");
        trie.addKeyword("Y");
        trie.addKeyword("XD");
    }

    public static void parsePattern(List<NR> nrList, List<Vertex> pWordSegResult, WordNet graphOptimum)
    {
        ListIterator<Vertex> listIterator = pWordSegResult.listIterator();
        StringBuilder sbPattern = new StringBuilder(nrList.size());
        NR preNR = NR.A;
        for (NR nr : nrList)
        {
            Vertex current = listIterator.next();
            switch (nr)
            {
                case U:
                    sbPattern.append(NR.K.toString());
                    sbPattern.append(NR.B.toString());
                    preNR = B;
                    listIterator.previous();
                    String nowK = current.realWord.substring(0, current.realWord.length() - 1);
                    String nowB = current.realWord.substring(current.realWord.length() - 1);
                    listIterator.set(new Vertex(nowK, new CoreDictionary.Attribute(0)));
                    listIterator.next();
                    listIterator.add(new Vertex(nowB, new CoreDictionary.Attribute(0)));
                    continue;
                case V:
                    if (preNR == B)
                    {
                        sbPattern.append(NR.E.toString());  //BE
                    }
                    else
                    {
                        sbPattern.append(NR.D.toString());  //CD
                    }
                    sbPattern.append(NR.L.toString());
                    preNR = L;
                    // 对串也做一些修改
                    listIterator.previous();
                    String nowED = current.realWord.substring(current.realWord.length() - 1);
                    String nowL = current.realWord.substring(0, current.realWord.length() - 1);
                    listIterator.set(new Vertex(nowED, new CoreDictionary.Attribute(0)));
                    listIterator.add(new Vertex(nowL, new CoreDictionary.Attribute(0)));
                    continue;
                default:
                    sbPattern.append(nr.toString());
                    break;
            }
            preNR = nr;
        }
        String pattern = sbPattern.toString();
        logger.trace("模式串：{}", pattern);
        logger.trace("对应串：{}", pWordSegResult);
        Collection<Emit> emitCollection = trie.parseText(pattern);
        Vertex[] wordArray = pWordSegResult.toArray(new Vertex[0]);
        int startMax = -1;
        for (Emit emit : emitCollection)
        {
            String keyword = emit.getKeyword();
            logger.trace("匹配到：{}", keyword);
            int start = emit.getStart();
            int end = emit.getEnd();
            StringBuilder sbName = new StringBuilder();
            for (int i = start; i <= end; ++i)
            {
                sbName.append(wordArray[i].realWord);
            }
            String name = sbName.toString();
            logger.trace("识别出：{}", name);
            int offset = 0;
            for (int i = 0; i < start; ++i)
            {
                offset += wordArray[i].realWord.length();
            }
            graphOptimum.add(offset, Vertex.newPersonInstance(name));
            startMax = Math.max(startMax, start);
        }
        if (startMax > 0)
        {
            graphOptimum.addAll(pWordSegResult.subList(0, startMax));
        }
    }
}
