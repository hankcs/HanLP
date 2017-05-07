/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 12:27</create-date>
 *
 * <copyright file="WordNatureWeightScorer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dependency.model;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLLoader;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.io.IOUtil;

import java.util.Set;
import java.util.TreeSet;

/**
 * 生成模型打分器模型构建工具
 *
 * @author hankcs
 */
public class WordNatureWeightModelMaker
{
    public static boolean makeModel(String corpusLoadPath, String modelSavePath)
    {
        Set<String> posSet = new TreeSet<String>();
        DictionaryMaker dictionaryMaker = new DictionaryMaker();
        for (CoNLLSentence sentence : CoNLLLoader.loadSentenceList(corpusLoadPath))
        {
            for (CoNLLWord word : sentence.word)
            {
                addPair(word.NAME, word.HEAD.NAME, word.DEPREL, dictionaryMaker);
                addPair(word.NAME, wrapTag(word.HEAD.POSTAG ), word.DEPREL, dictionaryMaker);
                addPair(wrapTag(word.POSTAG), word.HEAD.NAME, word.DEPREL, dictionaryMaker);
                addPair(wrapTag(word.POSTAG), wrapTag(word.HEAD.POSTAG), word.DEPREL, dictionaryMaker);
                posSet.add(word.POSTAG);
            }
        }
        for (CoNLLSentence sentence : CoNLLLoader.loadSentenceList(corpusLoadPath))
        {
            for (CoNLLWord word : sentence.word)
            {
                addPair(word.NAME, word.HEAD.NAME, word.DEPREL, dictionaryMaker);
                addPair(word.NAME, wrapTag(word.HEAD.POSTAG ), word.DEPREL, dictionaryMaker);
                addPair(wrapTag(word.POSTAG), word.HEAD.NAME, word.DEPREL, dictionaryMaker);
                addPair(wrapTag(word.POSTAG), wrapTag(word.HEAD.POSTAG), word.DEPREL, dictionaryMaker);
                posSet.add(word.POSTAG);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (String pos : posSet)
        {
            sb.append("case \"" + pos + "\":\n");
        }
        IOUtil.saveTxt("data/model/dependency/pos-thu.txt", sb.toString());
        return dictionaryMaker.saveTxtTo(modelSavePath);
    }

    private static void addPair(String from, String to, String label, DictionaryMaker dictionaryMaker)
    {
        dictionaryMaker.add(new Word(from + "@" + to, label));
        dictionaryMaker.add(new Word(from + "@", "频次"));
    }

    /**
     * 用尖括号将标签包起来
     * @param tag
     * @return
     */
    public static String wrapTag(String tag)
    {
        return "<" + tag + ">";
    }
}
