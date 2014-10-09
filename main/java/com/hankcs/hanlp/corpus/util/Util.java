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


import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

/**
 * @author hankcs
 */
public class Util
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
     * @param word
     * @return
     */
    public static IWord compile(IWord word)
    {
        switch (word.getLabel())
        {
            case "nr":
                return new Word(word.getValue(), TAG_PEOPLE);
            case "m":
            case "mq":
                return new Word(word.getValue(), TAG_NUMBER);
            case "t":
                return new Word(word.getValue(), TAG_TIME);
            case "ns":
                return new Word(word.getValue(), TAG_TIME);
        }

        return word;
    }
}
