/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/11 7:48</create-date>
 *
 * <copyright file="TestDependencyCorpus.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLLoader;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.LinkedList;

/**
 * @author hankcs
 */
public class TestDependencyCorpus extends TestCase
{
    public void testConvert() throws Exception
    {
        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\dev.conll.fixed.txt");
    }

    /**
     * 细粒度转粗粒度
     * @throws Exception
     */
    public void testPosTag() throws Exception
    {
        DictionaryMaker dictionaryMaker = new DictionaryMaker();
        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\dev.conll.fixed.txt");
        for (CoNLLSentence coNLLSentence : coNLLSentences)
        {
            for (CoNLLWord coNLLWord : coNLLSentence.word)
            {
                dictionaryMaker.add(new Item(coNLLWord.POSTAG, coNLLWord.CPOSTAG));
            }
        }
        System.out.println(dictionaryMaker.entrySet());
    }

    /**
     * 导出CRF训练语料
     * @throws Exception
     */
    public void testMakeCRF() throws Exception
    {
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Tools\\CRF++-0.58\\example\\dependency\\train.txt")));
        LinkedList<CoNLLSentence> coNLLSentences = CoNLLLoader.loadSentenceList("D:\\Doc\\语料库\\依存分析训练数据\\THU\\train.conll.fixed.txt");
        for (CoNLLSentence coNLLSentence : coNLLSentences)
        {
            for (CoNLLWord coNLLWord : coNLLSentence.word)
            {
                bw.write(coNLLWord.NAME);
                bw.write('\t');
                bw.write(coNLLWord.CPOSTAG);
                bw.write('\t');
                bw.write(coNLLWord.POSTAG);
                bw.write('\t');
                int d = coNLLWord.HEAD.ID - coNLLWord.ID;
                int posDistance = 1;
                if (d > 0)                          // 在后面
                {
                    for (int i = 1; i < d; ++i)
                    {
                        if (coNLLSentence.word[coNLLWord.ID - 1 + i].CPOSTAG.equals(coNLLWord.HEAD.CPOSTAG))
                        {
                            ++posDistance;
                        }
                    }
                }
                else
                {
                    for (int i = 1; i < -d; ++i)    // 在前面
                    {
                        if (coNLLSentence.word[coNLLWord.ID - 1 - i].CPOSTAG.equals(coNLLWord.HEAD.CPOSTAG))
                        {
                            ++posDistance;
                        }
                    }
                }
                bw.write((d > 0 ? "+" : "-") + posDistance + "_" + coNLLWord.HEAD.CPOSTAG
//                                 + "_" + coNLLWord.DEPREL
                );
                bw.newLine();
            }
            bw.newLine();
        }
        bw.close();
    }
}
