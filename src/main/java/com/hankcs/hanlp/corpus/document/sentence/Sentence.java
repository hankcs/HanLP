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

import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.WordFactory;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 句子，指的是以。，：！结尾的句子
 * @author hankcs
 */
public class Sentence implements Serializable
{
    public List<IWord> wordList;

    public Sentence(List<IWord> wordList)
    {
        this.wordList = wordList;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        for (IWord word : wordList)
        {
            sb.append(word);
            if (i != wordList.size()) sb.append(' ');
            ++i;
        }
        return sb.toString();
    }

    public static Sentence create(String param)
    {
        Pattern pattern = Pattern.compile("(\\[(([^\\s]+/[0-9a-zA-Z]+)\\s+)+?([^\\s]+/[0-9a-zA-Z]+)]/[0-9a-zA-Z]+)|([^\\s]+/[0-9a-zA-Z]+)");
        Matcher matcher = pattern.matcher(param);
        List<IWord> wordList = new LinkedList<IWord>();
        while (matcher.find())
        {
            String single = matcher.group();
            IWord word = WordFactory.create(single);
            if (word == null)
            {
                logger.warning("在用" + single + "构造单词时失败");
                return null;
            }
            wordList.add(word);
        }

        return new Sentence(wordList);
    }
}
