/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 17:42</create-date>
 *
 * <copyright file="CompoundWord.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document.sentence.word;


import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 复合词，由两个或以上的word构成
 * @author hankcs
 */
public class CompoundWord implements IWord, Iterable<Word>
{
    /**
     * 由这些词复合而来
     */
    public List<Word> innerList;

    /**
     * 标签，通常是词性
     */
    public String label;

    @Override
    public String getValue()
    {
        StringBuilder sb = new StringBuilder();
        for (Word word : innerList)
        {
            sb.append(word.value);
        }
        return sb.toString();
    }

    @Override
    public String getLabel()
    {
        return label;
    }

    @Override
    public void setLabel(String label)
    {
        this.label = label;
    }

    @Override
    public void setValue(String value)
    {
        innerList.clear();
        innerList.add(new Word(value, label));
    }

    @Override
    public int length()
    {
        return getValue().length();
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        int i = 1;
        for (Word word : innerList)
        {
            sb.append(word.toString());
            if (i != innerList.size())
            {
                sb.append(' ');
            }
            ++i;
        }
        sb.append("]/");
        sb.append(label);
        return sb.toString();
    }

    /**
     * 转换为一个简单词
     * @return
     */
    public Word toWord()
    {
        return new Word(getValue(), getLabel());
    }

    public CompoundWord(List<Word> innerList, String label)
    {
        this.innerList = innerList;
        this.label = label;
    }

    public static CompoundWord create(String param)
    {
        if (param == null) return null;
        int cutIndex = param.lastIndexOf(']');
        if (cutIndex <= 2 || cutIndex == param.length() - 1) return null;
        String wordParam  = param.substring(1, cutIndex);
        List<Word> wordList = new LinkedList<Word>();
        for (String single : wordParam.split("\\s+"))
        {
            if (single.length() == 0) continue;
            Word word = Word.create(single);
            if (word == null)
            {
                logger.warning("使用参数" + single + "构造单词时发生错误");
                return null;
            }
            wordList.add(word);
        }
        String labelParam = param.substring(cutIndex + 1);
        if (labelParam.startsWith("/"))
        {
            labelParam = labelParam.substring(1);
        }
        return new CompoundWord(wordList, labelParam);
    }

    @Override
    public Iterator<Word> iterator()
    {
        return innerList.iterator();
    }
}
