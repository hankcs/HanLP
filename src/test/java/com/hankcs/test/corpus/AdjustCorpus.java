/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 1:05</create-date>
 *
 * <copyright file="AdjustCorpus.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;


import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.EasyDictionary;
import com.hankcs.hanlp.corpus.dictionary.TFDictionary;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.document.CorpusLoader;
import com.hankcs.hanlp.corpus.document.Document;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.io.FolderWalker;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.occurrence.TermFrequency;
import com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.utility.Predefine;
import junit.framework.TestCase;

import java.io.*;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * 部分标注有问题，比如逗号缺少标注等等，尝试修复它
 * @author hankcs
 */
public class AdjustCorpus extends TestCase
{
    public void testAdjust() throws Exception
    {
        List<File> fileList = FolderWalker.open("D:\\JavaProjects\\CorpusToolBox\\data\\2014\\");
        for (File file : fileList)
        {
            handle(file);
        }
    }

    private static void handle(File file)
    {
        try
        {
            String text = IOUtil.readTxt(file.getPath());
            int length = text.length();
            text = addW(text, "：");
            text = addW(text, "？");
            text = addW(text, "，");
            text = addW(text, "）");
            text = addW(text, "（");
            text = addW(text, "！");
            text = addW(text, "(");
            text = addW(text, ")");
            text = addW(text, ",");
            text = addW(text, "‘");
            text = addW(text, "’");
            text = addW(text, "“");
            text = addW(text, "”");
            text = addW(text, ";");
            text = addW(text, "……");
            text = addW(text, "。");
            text = addW(text, "、");
            text = addW(text, "《");
            text = addW(text, "》");
            if (text.length() != length)
            {
                BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)));
                bw.write(text);
                bw.close();
                System.out.println("修正了" + file);
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private static String addW(String text, String c)
    {
        text = text.replaceAll("\\" + c + "/w ", c);
        return text.replaceAll("\\" + c, c + "/w ");
    }

    public void testPlay() throws Exception
    {
        final TFDictionary tfDictionary = new TFDictionary();
        CorpusLoader.walk("D:\\JavaProjects\\CorpusToolBox\\data\\2014", new CorpusLoader.Handler()
        {
            @Override
            public void handle(Document document)
            {
                for (List<IWord> wordList : document.getComplexSentenceList())
                {
                    for (IWord word : wordList)
                    {
                        if (word instanceof CompoundWord && word.getLabel().equals("ns"))
                        {
                            tfDictionary.add(word.toString());
                        }
                    }
                }
            }
        });
        tfDictionary.saveTxtTo("data/test/complex_ns.txt");
    }

    public void testAdjustNGram() throws Exception
    {
        IOUtil.LineIterator iterator = new IOUtil.LineIterator(HanLP.Config.BiGramDictionaryPath);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(HanLP.Config.BiGramDictionaryPath + "adjust.txt"), "UTF-8"));
        while (iterator.hasNext())
        {
            String line = iterator.next();
            String[] params = line.split(" ");
            String first = params[0].split("@", 2)[0];
            String second = params[0].split("@", 2)[1];
//            if (params.length != 2)
//                System.err.println(line);
            int biFrequency = Integer.parseInt(params[1]);
            CoreDictionary.Attribute attribute = CoreDictionary.get(first + second);
            if (attribute != null && (first.length() == 1 || second.length() == 1))
            {
                System.out.println(line);
                continue;
            }
            bw.write(line);
            bw.newLine();
        }
        bw.close();
    }

    public void testRemoveLabelD() throws Exception
    {
        Set<String> nameFollowers = new TreeSet<String>();
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(HanLP.Config.BiGramDictionaryPath);
        while (lineIterator.hasNext())
        {
            String line = lineIterator.next();
            String[] words = line.split("\\s")[0].split("@");
            if (words[0].equals(Predefine.TAG_PEOPLE))
            {
                nameFollowers.add(words[1]);
            }
        }
        DictionaryMaker dictionary = DictionaryMaker.load(HanLP.Config.PersonDictionaryPath);
        for (Map.Entry<String, Item> entry : dictionary.entrySet())
        {
            String key = entry.getKey();
            int dF = entry.getValue().getFrequency("D");
            if (key.length() == 1 && 0 < dF && dF < 100)
            {
                CoreDictionary.Attribute attribute = CoreDictionary.get(key);
                if (nameFollowers.contains(key)
                    || (attribute != null && attribute.hasNatureStartsWith("v") && attribute.totalFrequency > 1000)
                    )
                {
                    System.out.println(key);
                    entry.getValue().removeLabel("D");
                }
            }
        }

        dictionary.saveTxtTo(HanLP.Config.PersonDictionaryPath);
    }
}
