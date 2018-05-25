/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午7:41</create-date>
 *
 * <copyright file="LexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer.lexical;

import com.hankcs.hanlp.corpus.document.sentence.Sentence;

/**
 * @author hankcs
 */
public interface LexicalAnalyzer extends Segmenter, POSTagger, NERecognizer
{
    /**
     * 对句子进行词法分析
     *
     * @param sentence 纯文本句子
     * @return HanLP定义的结构化句子
     */
    Sentence analyze(final String sentence);
}
