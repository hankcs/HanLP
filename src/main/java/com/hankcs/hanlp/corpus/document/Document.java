/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/8 19:01</create-date>
 *
 * <copyright file="Document.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.document;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * @author hankcs
 */
public class Document implements Serializable
{
    public List<Sentence> sentenceList;

    public Document(List<Sentence> sentenceList)
    {
        this.sentenceList = sentenceList;
    }

    public static Document create(String param)
    {
        Pattern pattern = Pattern.compile(".+?((。/w)|(！/w )|(？/w )|\\n|$)");
        Matcher matcher = pattern.matcher(param);
        List<Sentence> sentenceList = new LinkedList<Sentence>();
        while (matcher.find())
        {
            String single = matcher.group();
            Sentence sentence = Sentence.create(single);
            if (sentence == null)
            {
                logger.warning("使用" + single + "构建句子失败");
                return null;
            }
            sentenceList.add(sentence);
        }
        return new Document(sentenceList);
    }

    /**
     * 获取单词序列
     *
     * @return
     */
    public List<IWord> getWordList()
    {
        List<IWord> wordList = new LinkedList<IWord>();
        for (Sentence sentence : sentenceList)
        {
            wordList.addAll(sentence.wordList);
        }
        return wordList;
    }

    public List<Word> getSimpleWordList()
    {
        List<IWord> wordList = getWordList();
        List<Word> simpleWordList = new LinkedList<Word>();
        for (IWord word : wordList)
        {
            if (word instanceof CompoundWord)
            {
                simpleWordList.addAll(((CompoundWord) word).innerList);
            }
            else
            {
                simpleWordList.add((Word) word);
            }
        }

        return simpleWordList;
    }

    /**
     * 获取简单的句子列表，其中复合词会被拆分为简单词
     * @return
     */
    public List<List<Word>> getSimpleSentenceList()
    {
        List<List<Word>> simpleList = new LinkedList<List<Word>>();
        for (Sentence sentence : sentenceList)
        {
            List<Word> wordList = new LinkedList<Word>();
            for (IWord word : sentence.wordList)
            {
                if (word instanceof CompoundWord)
                {
                    for (Word inner : ((CompoundWord) word).innerList)
                    {
                        wordList.add(inner);
                    }
                }
                else
                {
                    wordList.add((Word) word);
                }
            }
            simpleList.add(wordList);
        }

        return simpleList;
    }

    /**
     * 获取复杂句子列表，句子中的每个单词有可能是复合词，有可能是简单词
     * @return
     */
    public List<List<IWord>> getComplexSentenceList()
    {
        List<List<IWord>> complexList = new LinkedList<List<IWord>>();
        for (Sentence sentence : sentenceList)
        {
            complexList.add(sentence.wordList);
        }

        return complexList;
    }

    /**
     * 获取简单的句子列表
     * @param spilt 如果为真，其中复合词会被拆分为简单词
     * @return
     */
    public List<List<Word>> getSimpleSentenceList(boolean spilt)
    {
        List<List<Word>> simpleList = new LinkedList<List<Word>>();
        for (Sentence sentence : sentenceList)
        {
            List<Word> wordList = new LinkedList<Word>();
            for (IWord word : sentence.wordList)
            {
                if (word instanceof CompoundWord)
                {
                    if (spilt)
                    {
                        for (Word inner : ((CompoundWord) word).innerList)
                        {
                            wordList.add(inner);
                        }
                    }
                    else
                    {
                        wordList.add(((CompoundWord) word).toWord());
                    }
                }
                else
                {
                    wordList.add((Word) word);
                }
            }
            simpleList.add(wordList);
        }

        return simpleList;
    }

    /**
     * 获取简单的句子列表，其中复合词的标签如果是set中指定的话会被拆分为简单词
     * @param labelSet
     * @return
     */
    public List<List<Word>> getSimpleSentenceList(Set<String> labelSet)
    {
        List<List<Word>> simpleList = new LinkedList<List<Word>>();
        for (Sentence sentence : sentenceList)
        {
            List<Word> wordList = new LinkedList<Word>();
            for (IWord word : sentence.wordList)
            {
                if (word instanceof CompoundWord)
                {
                    if (labelSet.contains(word.getLabel()))
                    {
                        for (Word inner : ((CompoundWord) word).innerList)
                        {
                            wordList.add(inner);
                        }
                    }
                    else
                    {
                        wordList.add(((CompoundWord) word).toWord());
                    }
                }
                else
                {
                    wordList.add((Word) word);
                }
            }
            simpleList.add(wordList);
        }

        return simpleList;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for (Sentence sentence : sentenceList)
        {
            sb.append(sentence);
            sb.append(' ');
        }
        if (sb.length() > 0) sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }
}
