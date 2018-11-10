/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-11-10 10:23 AM</create-date>
 *
 * <copyright file="PipelineLexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * See LICENSE file in the project root for full license information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.pipe;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.tokenizer.lexical.LexicalAnalyzer;

import java.util.LinkedList;
import java.util.List;

/**
 * 流水线式词法分析器
 * @author hankcs
 */
public class LexicalAnalyzerPipeline extends Pipeline<String, List<IWord>, List<IWord>> implements LexicalAnalyzer
{
    public LexicalAnalyzerPipeline(Pipe<String, List<IWord>> first, Pipe<List<IWord>, List<IWord>> last)
    {
        super(first, last);
    }

    public LexicalAnalyzerPipeline(LexicalAnalyzer analyzer)
    {
        this(new LexicalAnalyzerPipe(analyzer));
    }

    public LexicalAnalyzerPipeline(LexicalAnalyzerPipe analyzer)
    {
        this(new Pipe<String, List<IWord>>()
             {
                 @Override
                 public List<IWord> flow(String input)
                 {
                     List<IWord> output = new LinkedList<IWord>();
                     output.add(new Word(input, null));
                     return output;
                 }
             },
             new Pipe<List<IWord>, List<IWord>>()
             {
                 @Override
                 public List<IWord> flow(List<IWord> input)
                 {
                     return input;
                 }
             }
        );
        add(analyzer);
    }

    /**
     * 获取代理的词法分析器
     *
     * @return
     */
    public LexicalAnalyzer getAnalyzer()
    {
        for (Pipe<List<IWord>, List<IWord>> pipe : this)
        {
            if (pipe instanceof LexicalAnalyzerPipe)
            {
                return ((LexicalAnalyzerPipe) pipe).analyzer;
            }
        }
        return null;
    }

    @Override
    public void segment(String sentence, String normalized, List<String> wordList)
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        analyzer.segment(sentence, normalized, wordList);
    }

    @Override
    public List<String> segment(String sentence)
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        return analyzer.segment(sentence);
    }

    @Override
    public String[] recognize(String[] wordArray, String[] posArray)
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        return analyzer.recognize(wordArray, posArray);
    }

    @Override
    public String[] tag(String... words)
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        return analyzer.tag(words);
    }

    @Override
    public String[] tag(List<String> wordList)
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        return analyzer.tag(wordList);
    }

    @Override
    public NERTagSet getNERTagSet()
    {
        LexicalAnalyzer analyzer = getAnalyzer();
        if (analyzer == null)
            throw new IllegalStateException("流水线中没有LexicalAnalyzerPipe");
        return analyzer.getNERTagSet();
    }

    @Override
    public Sentence analyze(String sentence)
    {
        return new Sentence(flow(sentence));
    }
}
