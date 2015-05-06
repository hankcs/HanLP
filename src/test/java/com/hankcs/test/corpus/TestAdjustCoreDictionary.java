/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/24 12:11</create-date>
 *
 * <copyright file="TestAdjustCoreDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.TFDictionary;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.occurrence.TermFrequency;
import com.hankcs.hanlp.corpus.util.CorpusUtil;
import junit.framework.TestCase;

import java.util.List;
import java.util.Map;

/**
 * 往核心词典里补充等效词串
 * @author hankcs
 */
public class TestAdjustCoreDictionary extends TestCase
{

    public static final String DATA_DICTIONARY_CORE_NATURE_DICTIONARY_TXT = "data/dictionary/CoreNatureDictionary.txt";

    public void testGetCompiledWordFromDictionary() throws Exception
    {
        DictionaryMaker dictionaryMaker = DictionaryMaker.load("data/test/CoreNatureDictionary.txt");
        for (Map.Entry<String, Item> entry : dictionaryMaker.entrySet())
        {
            String word = entry.getKey();
            Item item = entry.getValue();
            if (word.matches(".##."))
            {
                System.out.println(item);
            }
        }
    }

    public void testViewNGramDictionary() throws Exception
    {
        TFDictionary tfDictionary = new TFDictionary();
        tfDictionary.load("data/dictionary/CoreNatureDictionary.ngram.txt");
        for (Map.Entry<String, TermFrequency> entry : tfDictionary.entrySet())
        {
            String word = entry.getKey();
            TermFrequency frequency = entry.getValue();
            if (word.contains("##"))
            {
                System.out.println(frequency);
            }
        }
    }

    public void testSortCoreNatureDictionary() throws Exception
    {
        DictionaryMaker dictionaryMaker = DictionaryMaker.load(DATA_DICTIONARY_CORE_NATURE_DICTIONARY_TXT);
        dictionaryMaker.saveTxtTo(DATA_DICTIONARY_CORE_NATURE_DICTIONARY_TXT);
    }

    public void testSimplifyNZ() throws Exception
    {
        final DictionaryMaker nzDictionary = new DictionaryMaker();
        CorpusLoader.walk("D:\\Doc\\语料库\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                for (List<IWord> sentence : document.getComplexSentenceList())
                {
                    for (IWord word : sentence)
                    {
                        if (word instanceof CompoundWord && "nz".equals(word.getLabel()))
                        {
                            nzDictionary.add(word);
                        }
                    }
                }
            }
        });
        nzDictionary.saveTxtTo("data/test/nz.txt");
    }
}
