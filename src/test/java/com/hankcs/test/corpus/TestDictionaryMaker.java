/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 0:38</create-date>
 *
 * <copyright file="TestDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import junit.framework.TestCase;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class TestDictionaryMaker extends TestCase
{
    public void testSingleDocument() throws Exception
    {
        Document document = CorpusLoader.convert2Document(new File("data/2014/0101/c1002-23996898.txt"));
        DictionaryMaker dictionaryMaker = new DictionaryMaker();
        System.out.println(document);
        addToDictionary(document, dictionaryMaker);
        dictionaryMaker.saveTxtTo("data/dictionaryTest.txt");
    }

    private void addToDictionary(Document document, DictionaryMaker dictionaryMaker)
    {
        for (IWord word : document.getWordList())
        {
            if (word instanceof CompoundWord)
            {
                for (Word inner : ((CompoundWord)word).innerList)
                {
                    // 暂时不统计人名
                    if (inner.getLabel().equals("nr"))
                    {
                        continue;
                    }
                    // 如果需要人名，注销上面这句即可
                    dictionaryMaker.add(inner);
                }
            }
            // 暂时不统计人名
            if (word.getLabel().equals("nr"))
            {
                continue;
            }
            // 如果需要人名，注销上面这句即可
            dictionaryMaker.add(word);
        }
    }

    public void testMakeDictionary() throws Exception
    {
        final DictionaryMaker dictionaryMaker = new DictionaryMaker();
        CorpusLoader.walk("data/2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                addToDictionary(document, dictionaryMaker);
            }
        });
        dictionaryMaker.saveTxtTo("data/2014_dictionary.txt");
    }

    public void testLoadItemList() throws Exception
    {
        List<Item> itemList = DictionaryMaker.loadAsItemList("data/2014_dictionary.txt");
        Map<String, Integer> labelMap = new TreeMap<String, Integer>();
        for (Item item : itemList)
        {
            for (Map.Entry<String, Integer> entry : item.labelMap.entrySet())
            {
                Integer frequency = labelMap.get(entry.getKey());
                if (frequency == null) frequency = 0;
                labelMap.put(entry.getKey(), frequency + entry.getValue());
            }
        }
        for (String label : labelMap.keySet())
        {
            System.out.println(label);
        }
        System.out.println(labelMap.size());
    }

    public void testLoadEasyDictionary() throws Exception
    {
        EasyDictionary dictionary = EasyDictionary.create("data/2014_dictionary.txt");
        System.out.println(dictionary.GetWordInfo("高峰"));
    }

}
