/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-03 12:44 PM</create-date>
 *
 * <copyright file="HMMLexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.hmm;

import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;

/**
 * 基于隐马尔可夫模型的词法分析器
 *
 * @author hankcs
 */
public class HMMLexicalAnalyzer extends AbstractLexicalAnalyzer
{
    private HMMLexicalAnalyzer()
    {
    }

    public HMMLexicalAnalyzer(HMMSegmenter segmenter)
    {
        super(segmenter);
    }

    public HMMLexicalAnalyzer(HMMSegmenter segmenter, HMMPOSTagger posTagger)
    {
        super(segmenter, posTagger);
    }

    public HMMLexicalAnalyzer(HMMSegmenter segmenter, HMMPOSTagger posTagger, HMMNERecognizer neRecognizer)
    {
        super(segmenter, posTagger, neRecognizer);
    }
}
