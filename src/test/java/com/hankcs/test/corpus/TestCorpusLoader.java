/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 15:08</create-date>
 *
 * <copyright file="TestCorpusLoader.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;


/**
 * @author hankcs
 */
public class TestCorpusLoader extends TestCase
{
    public void testMultiThread() throws Exception
    {
        CorpusLoader.HandlerThread[] handlerThreadArray = new CorpusLoader.HandlerThread[4];
        for (int i = 0; i < handlerThreadArray.length; ++i)
        {
            handlerThreadArray[i] = new CorpusLoader.HandlerThread(String.valueOf(i))
            {
                @Override
                public void handle(Document document)
                {

                }
            };
        }
        CorpusLoader.walk("data/2014", handlerThreadArray);
    }

    public void testSingleThread() throws Exception
    {
        CorpusLoader.walk("data/2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {

            }
        });
    }

    public void testConvert2SimpleSentenceList() throws Exception
    {
        List<List<Word>> simpleSentenceList = CorpusLoader.convert2SimpleSentenceList("data/2014");
        System.out.println(simpleSentenceList.get(0));
    }

    public void testMakePersonCustomDictionary() throws Exception
    {
        final DictionaryMaker dictionaryMaker = new DictionaryMaker();
        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                List<List<IWord>> complexSentenceList = document.getComplexSentenceList();
                for (List<IWord> wordList : complexSentenceList)
                {
                    for (IWord word : wordList)
                    {
                        if (word.getLabel().startsWith("nr"))
                        {
                            dictionaryMaker.add(word);
                        }
                    }
                }
            }
        });
        dictionaryMaker.saveTxtTo("data/dictionary/custom/人名词典.txt");
    }

    public void testMakeOrganizationCustomDictionary() throws Exception
    {
        final DictionaryMaker dictionaryMaker = new DictionaryMaker();
        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                List<List<IWord>> complexSentenceList = document.getComplexSentenceList();
                for (List<IWord> wordList : complexSentenceList)
                {
                    for (IWord word : wordList)
                    {
                        if (word.getLabel().startsWith("nt"))
                        {
                            dictionaryMaker.add(word);
                        }
                    }
                }
            }
        });
        dictionaryMaker.saveTxtTo("data/dictionary/custom/机构名词典.txt");
    }
}
