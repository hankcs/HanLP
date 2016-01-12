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
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.*;
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

    public void testCombineToTxt() throws Exception
    {
        final BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Doc\\语料库\\2014_cn.txt"), "UTF-8"));
        CorpusLoader.walk("D:\\Doc\\语料库\\2014_hankcs", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                try
                {
                    for (List<Word> sentence : document.getSimpleSentenceList())
                    {
                        for (IWord word : sentence)
                        {
                            bw.write(word.getValue());
                            bw.write(' ');
                        }
                        bw.newLine();
                    }
                    bw.newLine();
                }
                catch (Exception e)
                {
                    e.printStackTrace();
                }
            }
            });
        bw.close();
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

    /**
     * 语料库中有很多句号标注得不对，尝试纠正它们
     * 比如“方言/n 版/n [新年/t 祝福/vn]/nz 。你/rr 的/ude1 一段/mq 话/n ”
     * @throws Exception
     */
    public void testAdjustDot() throws Exception
    {
        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
        {
            int id = 0;
            @Override
            public void handle(Document document)
            {
                try
                {
                    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("D:\\Doc\\语料库\\2014_hankcs\\" + (++id) + ".txt"), "UTF-8"));
                    for (List<IWord> wordList : document.getComplexSentenceList())
                    {
                        if (wordList.size() == 0) continue;
                        for (IWord word : wordList)
                        {
                            if (word.getValue().length() > 1 && word.getValue().charAt(0) == '。')
                            {
                                bw.write("。/w");
                                bw.write(word.getValue().substring(1));
                                bw.write('/');
                                bw.write(word.getLabel());
                                bw.write(' ');
                                continue;
                            }
                            bw.write(word.toString());
                            bw.write(' ');
                        }
                        bw.newLine();
                    }
                    bw.close();
                }
                catch (FileNotFoundException e)
                {
                    e.printStackTrace();
                }
                catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                }
            }
        });
    }

    public void testLoadMyCorpus() throws Exception
    {
        CorpusLoader.walk("D:\\Doc\\语料库\\2014_hankcs\\", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                for (List<IWord> wordList : document.getComplexSentenceList())
                {
                    System.out.println(wordList);
                }
            }
        });

    }

    /**
     * 有些引号不对
     * @throws Exception
     */
    public void testFindQuote() throws Exception
    {
        CorpusLoader.walk("D:\\Doc\\语料库\\2014_hankcs\\", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                for (List<Word> wordList : document.getSimpleSentenceList())
                {
                    for (Word word : wordList)
                    {
                        if(word.value.length() > 1 && word.value.endsWith("\""))
                        {
                            System.out.println(word);
                        }
                    }
                }
            }
        });
    }
}
