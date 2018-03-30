/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 18:04</create-date>
 *
 * <copyright file="Sentence.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document.sentence;

import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.document.sentence.word.WordFactory;
import com.hankcs.hanlp.dictionary.other.PartOfSpeechTagDictionary;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 句子，指的是以。！等标点结尾的句子
 *
 * @author hankcs
 */
public class Sentence implements Serializable, Iterable<IWord>
{
    /**
     * 词语列表（复合或简单单词的列表）
     */
    public List<IWord> wordList;

    public Sentence(List<IWord> wordList)
    {
        this.wordList = wordList;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder(size() * 4);
        int i = 1;
        for (IWord word : wordList)
        {
            sb.append(word);
            if (i != wordList.size()) sb.append(' ');
            ++i;
        }
        return sb.toString();
    }

    /**
     * brat standoff format<br>
     * http://brat.nlplab.org/standoff.html
     *
     * @return
     */
    public String toStandoff()
    {
        StringBuilder sb = new StringBuilder(size() * 4);
        String delimiter = " ";
        String text = text(delimiter);
        sb.append(text).append('\n');
        int i = 1;
        int offset = 0;
        for (IWord word : wordList)
        {
            assert text.charAt(offset) == word.getValue().charAt(0);
            printWord(word, sb, i, offset);
            ++i;
            if (word instanceof CompoundWord)
            {
                int offsetChild = offset;
                for (Word child : ((CompoundWord) word).innerList)
                {
                    printWord(child, sb, i, offsetChild);
                    offsetChild += child.length();
                    offsetChild += delimiter.length();
                    ++i;
                }
                offset += delimiter.length() * ((CompoundWord) word).innerList.size();
            }
            else
            {
                offset += delimiter.length();
            }
            offset += word.length();
        }
        return sb.toString();
    }

    /**
     * 按照 PartOfSpeechTagDictionary 指定的映射表将词语词性翻译过去
     *
     * @return
     */
    public Sentence translateLabels()
    {
        for (IWord word : wordList)
        {
            word.setLabel(PartOfSpeechTagDictionary.translate(word.getLabel()));
            if (word instanceof CompoundWord)
            {
                for (Word child : ((CompoundWord) word).innerList)
                {
                    child.setLabel(PartOfSpeechTagDictionary.translate(child.getLabel()));
                }
            }
        }
        return this;
    }

    /**
     * 按照 PartOfSpeechTagDictionary 指定的映射表将复合词词语词性翻译过去
     *
     * @return
     */
    public Sentence translateCompoundWordLabels()
    {
        for (IWord word : wordList)
        {
            if (word instanceof CompoundWord)
                word.setLabel(PartOfSpeechTagDictionary.translate(word.getLabel()));
        }
        return this;
    }

    private void printWord(IWord word, StringBuilder sb, int id, int offset)
    {
        char delimiter = '\t';
        char endLine = '\n';
        sb.append('T').append(id).append(delimiter);
        sb.append(word.getLabel()).append(delimiter);
        int length = word.length();
        if (word instanceof CompoundWord)
        {
            length += ((CompoundWord) word).innerList.size() - 1;
        }
        sb.append(offset).append(delimiter).append(offset + length).append(delimiter);
        sb.append(word.getValue()).append(endLine);
    }

    /**
     * 以人民日报2014语料格式的字符串创建一个结构化句子
     *
     * @param param
     * @return
     */
    public static Sentence create(String param)
    {
        if (param == null)
        {
            return null;
        }
        param = param.trim();
        if (param.isEmpty())
        {
            return null;
        }
        Pattern pattern = Pattern.compile("(\\[(([^\\s]+/[0-9a-zA-Z]+)\\s+)+?([^\\s]+/[0-9a-zA-Z]+)]/?[0-9a-zA-Z]+)|([^\\s]+/[0-9a-zA-Z]+)");
        Matcher matcher = pattern.matcher(param);
        List<IWord> wordList = new LinkedList<IWord>();
        while (matcher.find())
        {
            String single = matcher.group();
            IWord word = WordFactory.create(single);
            if (word == null)
            {
                logger.warning("在用 " + single + " 构造单词时失败，句子构造参数为 " + param);
                return null;
            }
            wordList.add(word);
        }
        if (wordList.isEmpty()) // 按照无词性来解析
        {
            for (String w : param.split("\\s+"))
            {
                wordList.add(new Word(w, null));
            }
        }

        return new Sentence(wordList);
    }

    /**
     * 句子中单词（复合词或简单词）的数量
     *
     * @return
     */
    public int size()
    {
        return wordList.size();
    }

    /**
     * 句子文本长度
     *
     * @return
     */
    public int length()
    {
        int length = 0;
        for (IWord word : this)
        {
            length += word.getValue().length();
        }

        return length;
    }

    /**
     * 原始文本形式（无标注，raw text）
     *
     * @return
     */
    public String text()
    {
        return text(null);
    }

    /**
     * 原始文本形式（无标注，raw text）
     *
     * @param delimiter 词语之间的分隔符
     * @return
     */
    public String text(String delimiter)
    {
        if (delimiter == null) delimiter = "";
        StringBuilder sb = new StringBuilder(size() * 3);
        for (IWord word : this)
        {
            if (word instanceof CompoundWord)
            {
                for (Word child : ((CompoundWord) word).innerList)
                {
                    sb.append(child.getValue()).append(delimiter);
                }
            }
            else
            {
                sb.append(word.getValue()).append(delimiter);
            }
        }
        sb.setLength(sb.length() - delimiter.length());

        return sb.toString();
    }

    @Override
    public Iterator<IWord> iterator()
    {
        return wordList.iterator();
    }

    /**
     * 找出所有词性为label的单词（不检查复合词内部的简单词）
     *
     * @param label
     * @return
     */
    public List<IWord> findWordsByLabel(String label)
    {
        List<IWord> wordList = new LinkedList<IWord>();
        for (IWord word : this)
        {
            if (label.equals(word.getLabel()))
            {
                wordList.add(word);
            }
        }
        return wordList;
    }

    /**
     * 找出第一个词性为label的单词（不检查复合词内部的简单词）
     *
     * @param label
     * @return
     */
    public IWord findFirstWordByLabel(String label)
    {
        for (IWord word : this)
        {
            if (label.equals(word.getLabel()))
            {
                return word;
            }
        }
        return null;
    }

    /**
     * 找出第一个词性为label的单词的指针（不检查复合词内部的简单词）<br>
     * 若要查看该单词，请调用 previous<br>
     * 若要删除该单词，请调用 remove<br>
     *
     * @param label
     * @return
     */
    public ListIterator<IWord> findFirstWordIteratorByLabel(String label)
    {
        ListIterator<IWord> listIterator = this.wordList.listIterator();
        while (listIterator.hasNext())
        {
            IWord word = listIterator.next();
            if (label.equals(word.getLabel()))
            {
                return listIterator;
            }
        }
        return null;
    }

    /**
     * 是否含有词性为label的单词
     *
     * @param label
     * @return
     */
    public boolean containsWordWithLabel(String label)
    {
        return findFirstWordByLabel(label) != null;
    }

    /**
     * 转换为简单单词列表
     *
     * @return
     */
    public List<Word> toSimpleWordList()
    {
        List<Word> wordList = new LinkedList<Word>();
        for (IWord word : this.wordList)
        {
            if (word instanceof CompoundWord)
            {
                wordList.addAll(((CompoundWord) word).innerList);
            }
            else
            {
                wordList.add((Word) word);
            }
        }

        return wordList;
    }
}
