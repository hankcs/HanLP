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
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.utility.Utility;

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
     * 转换为空格分割无标签的String
     *
     * @return
     */
    public String toStringWithoutLabels()
    {
        StringBuilder sb = new StringBuilder(size() * 4);
        int i = 1;
        for (IWord word : wordList)
        {
            if (word instanceof CompoundWord)
            {
                int j = 0;
                for (Word w : ((CompoundWord) word).innerList)
                {
                    sb.append(w.getValue());
                    if (++j != ((CompoundWord) word).innerList.size())
                        sb.append(' ');
                }
            }
            else
                sb.append(word.getValue());
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
        return toStandoff(false);
    }

    /**
     * brat standoff format<br>
     * http://brat.nlplab.org/standoff.html
     *
     * @param withComment
     * @return
     */
    public String toStandoff(boolean withComment)
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
            printWord(word, sb, i, offset, withComment);
            ++i;
            if (word instanceof CompoundWord)
            {
                int offsetChild = offset;
                for (Word child : ((CompoundWord) word).innerList)
                {
                    printWord(child, sb, i, offsetChild, withComment);
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
        printWord(word, sb, id, offset, false);
    }

    private void printWord(IWord word, StringBuilder sb, int id, int offset, boolean withComment)
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
        String translated = PartOfSpeechTagDictionary.translate(word.getLabel());
        if (withComment && !word.getLabel().equals(translated))
        {
            sb.append('#').append(id).append(delimiter).append("AnnotatorNotes").append(delimiter)
                .append('T').append(id).append(delimiter).append(translated)
                .append(endLine);
        }
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

    /**
     * 获取所有单词构成的数组
     *
     * @return
     */
    public String[] toWordArray()
    {
        List<Word> wordList = toSimpleWordList();
        String[] wordArray = new String[wordList.size()];
        Iterator<Word> iterator = wordList.iterator();
        for (int i = 0; i < wordArray.length; i++)
        {
            wordArray[i] = iterator.next().value;
        }
        return wordArray;
    }

    /**
     * word pos
     *
     * @return
     */
    public String[][] toWordTagArray()
    {
        List<Word> wordList = toSimpleWordList();
        String[][] pair = new String[2][wordList.size()];
        Iterator<Word> iterator = wordList.iterator();
        for (int i = 0; i < pair[0].length; i++)
        {
            Word word = iterator.next();
            pair[0][i] = word.value;
            pair[1][i] = word.label;
        }
        return pair;
    }

    /**
     * word pos ner
     *
     * @param tagSet
     * @return
     */
    public String[][] toWordTagNerArray(NERTagSet tagSet)
    {
        List<String[]> tupleList = Utility.convertSentenceToNER(this, tagSet);
        String[][] result = new String[3][tupleList.size()];
        Iterator<String[]> iterator = tupleList.iterator();
        for (int i = 0; i < result[0].length; i++)
        {
            String[] tuple = iterator.next();
            for (int j = 0; j < 3; ++j)
            {
                result[j][i] = tuple[j];
            }
        }
        return result;
    }

    public Sentence mergeCompoundWords()
    {
        ListIterator<IWord> listIterator = wordList.listIterator();
        while (listIterator.hasNext())
        {
            IWord word = listIterator.next();
            if (word instanceof CompoundWord)
            {
                listIterator.set(new Word(word.getValue(), word.getLabel()));
            }
        }
        return this;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Sentence sentence = (Sentence) o;
        return toString().equals(sentence.toString());
    }

    @Override
    public int hashCode()
    {
        return toString().hashCode();
    }
}
