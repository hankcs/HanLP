/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-11-10 10:36 AM</create-date>
 *
 * <copyright file="PipeLexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.pipe;

import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.tokenizer.lexical.LexicalAnalyzer;

import java.util.List;
import java.util.ListIterator;

/**
 * 词法分析器管道。约定将IWord的label设为非null表示本级管道已经处理
 *
 * @author hankcs
 */
public class LexicalAnalyzerPipe implements Pipe<List<IWord>, List<IWord>>
{
    /**
     * 代理的词法分析器
     */
    protected LexicalAnalyzer analyzer;

    public LexicalAnalyzerPipe(LexicalAnalyzer analyzer)
    {
        this.analyzer = analyzer;
    }

    @Override
    public List<IWord> flow(List<IWord> input)
    {
        ListIterator<IWord> listIterator = input.listIterator();
        while (listIterator.hasNext())
        {
            IWord wordOrSentence = listIterator.next();
            if (wordOrSentence.getLabel() != null)
                continue; // 这是别的管道已经处理过的单词，跳过
            listIterator.remove(); // 否则是句子
            String sentence = wordOrSentence.getValue();
            for (IWord word : analyzer.analyze(sentence))
            {
                listIterator.add(word);
            }
        }
        return input;
    }
}
