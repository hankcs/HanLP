/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-03-30 下午7:29</create-date>
 *
 * <copyright file="CRFLexicalAnalyzer.java">
 * Copyright (c) 2018, Han He. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;

import java.io.IOException;

/**
 * CRF词法分析器（中文分词、词性标注和命名实体识别）
 *
 * @author hankcs
 * @since 1.6.2
 */
public class CRFLexicalAnalyzer extends AbstractLexicalAnalyzer
{
    /**
     * 构造CRF词法分析器
     *
     * @param segmenter CRF分词器
     */
    public CRFLexicalAnalyzer(CRFSegmenter segmenter)
    {
        this.segmenter = segmenter;
    }

    /**
     * 构造CRF词法分析器
     *
     * @param segmenter CRF分词器
     * @param posTagger CRF词性标注器
     */
    public CRFLexicalAnalyzer(CRFSegmenter segmenter, CRFPOSTagger posTagger)
    {
        this.segmenter = segmenter;
        this.posTagger = posTagger;
        config.speechTagging = true;
    }

    /**
     * 构造CRF词法分析器
     *
     * @param segmenter    CRF分词器
     * @param posTagger    CRF词性标注器
     * @param neRecognizer CRF命名实体识别器
     */
    public CRFLexicalAnalyzer(CRFSegmenter segmenter, CRFPOSTagger posTagger, CRFNERecognizer neRecognizer)
    {
        this.segmenter = segmenter;
        this.posTagger = posTagger;
        this.neRecognizer = neRecognizer;
        config.speechTagging = true;
        config.nameRecognize = true;
    }

    /**
     * 构造CRF词法分析器
     *
     * @param cwsModelPath CRF分词器模型路径
     */
    public CRFLexicalAnalyzer(String cwsModelPath) throws IOException
    {
        this(new CRFSegmenter(cwsModelPath));
    }

    /**
     * 构造CRF词法分析器
     *
     * @param cwsModelPath CRF分词器模型路径
     * @param posModelPath CRF词性标注器模型路径
     */
    public CRFLexicalAnalyzer(String cwsModelPath, String posModelPath) throws IOException
    {
        this(new CRFSegmenter(cwsModelPath), new CRFPOSTagger(posModelPath));
    }

    /**
     * 构造CRF词法分析器
     *
     * @param cwsModelPath CRF分词器模型路径
     * @param posModelPath CRF词性标注器模型路径
     * @param nerModelPath CRF命名实体识别器模型路径
     */
    public CRFLexicalAnalyzer(String cwsModelPath, String posModelPath, String nerModelPath) throws IOException
    {
        this(new CRFSegmenter(cwsModelPath), new CRFPOSTagger(posModelPath), new CRFNERecognizer(nerModelPath));
    }

    /**
     * 加载配置文件指定的模型
     *
     * @throws IOException
     */
    public CRFLexicalAnalyzer() throws IOException
    {
        this(new CRFSegmenter(), new CRFPOSTagger(), new CRFNERecognizer());
    }
}
