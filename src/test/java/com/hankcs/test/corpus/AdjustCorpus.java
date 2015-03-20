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
import junit.framework.TestCase;

import java.io.*;
import java.util.List;

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
}
