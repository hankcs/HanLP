/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/7/14 11:01</create-date>
 *
 * <copyright file="WordNatureUtil.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.common.Term;

/**
 * 跟词语与词性有关的工具类，可以全局动态修改HanLP内部词库
 *
 * @author hankcs
 */
public class Lexicon
{
    /**
     * 从HanLP的词库中提取某个单词的属性（包括核心词典和用户词典）
     *
     * @param word 单词
     * @return 包含词性与频次的信息
     */
    public static CoreDictionary.Attribute getAttribute(String word)
    {
        CoreDictionary.Attribute attribute = CoreDictionary.get(word);
        if (attribute != null) return attribute;
        return CustomDictionary.get(word);
    }

    /**
     * 从HanLP的词库中提取某个单词的属性（包括核心词典和用户词典）
     *
     * @param term 单词
     * @return 包含词性与频次的信息
     */
    public static CoreDictionary.Attribute getAttribute(Term term)
    {
        return getAttribute(term.word);
    }

    /**
     * 获取某个单词的词频
     * @param word
     * @return
     */
    public static int getFrequency(String word)
    {
        CoreDictionary.Attribute attribute = getAttribute(word);
        if (attribute == null) return 0;
        return attribute.totalFrequency;
    }

    /**
     * 设置某个单词的属性
     * @param word
     * @param attribute
     * @return
     */
    public static boolean setAttribute(String word, CoreDictionary.Attribute attribute)
    {
        if (attribute == null) return false;

        if (CoreDictionary.trie.set(word, attribute)) return true;
        if (CustomDictionary.dat.set(word, attribute)) return true;
        CustomDictionary.trie.put(word, attribute);
        return true;
    }

    /**
     * 设置某个单词的属性
     * @param word
     * @param natureWithFrequency
     * @return
     */
    public static boolean setAttribute(String word, String natureWithFrequency)
    {
        CoreDictionary.Attribute attribute = CoreDictionary.Attribute.create(natureWithFrequency);
        return setAttribute(word, attribute);
    }
}
