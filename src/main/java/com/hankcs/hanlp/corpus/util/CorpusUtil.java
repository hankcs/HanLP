/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 21:15</create-date>
 *
 * <copyright file="Util.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.util;


import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * @author hankcs
 */
public class CorpusUtil
{
    public final static String TAG_PLACE = "未##地";
    public final static String TAG_BIGIN = "始##始";
    public final static String TAG_OTHER = "未##它";
    public final static String TAG_GROUP = "未##团";
    public final static String TAG_NUMBER = "未##数";
    public final static String TAG_PROPER = "未##专";
    public final static String TAG_TIME = "未##时";
    public final static String TAG_CLUSTER = "未##串";
    public final static String TAG_END = "末##末";
    public final static String TAG_PEOPLE = "未##人";

    /**
     * 编译单词
     *
     * @param word
     * @return
     */
    public static IWord compile(IWord word)
    {
        String label = word.getLabel();
        if ("nr".equals(label)) return new Word(word.getValue(), TAG_PEOPLE);
        else if ("m".equals(label) || "mq".equals(label)) return new Word(word.getValue(), TAG_NUMBER);
        else if ("t".equals(label)) return new Word(word.getValue(), TAG_TIME);
        else if ("ns".equals(label)) return new Word(word.getValue(), TAG_PLACE);
//        switch (word.getLabel())
//        {
//            case "nr":
//                return new Word(word.getValue(), TAG_PEOPLE);
//            case "m":
//            case "mq":
//                return new Word(word.getValue(), TAG_NUMBER);
//            case "t":
//                return new Word(word.getValue(), TAG_TIME);
//            case "ns":
//                return new Word(word.getValue(), TAG_TIME);
//        }

        return word;
    }

    /**
     * 将word列表转为兼容的IWord列表
     *
     * @param simpleSentenceList
     * @return
     */
    public static List<List<IWord>> convert2CompatibleList(List<List<Word>> simpleSentenceList)
    {
        List<List<IWord>> compatibleList = new LinkedList<List<IWord>>();
        for (List<Word> wordList : simpleSentenceList)
        {
            compatibleList.add(new LinkedList<IWord>(wordList));
        }
        return compatibleList;
    }

    public static List<IWord> spilt(List<IWord> wordList)
    {
        ListIterator<IWord> listIterator = wordList.listIterator();
        while (listIterator.hasNext())
        {
            IWord word = listIterator.next();
            if (word instanceof CompoundWord)
            {
                listIterator.remove();
                for (Word inner : ((CompoundWord) word).innerList)
                {
                    listIterator.add(inner);
                }
            }
        }
        return wordList;
    }
}
