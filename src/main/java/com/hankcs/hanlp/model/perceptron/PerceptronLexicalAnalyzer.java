/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-05 PM7:56</create-date>
 *
 * <copyright file="AveragedPerceptronSegment.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;

import java.io.IOException;
import java.util.List;

/**
 * 感知机词法分析器，支持简繁全半角和大小写
 *
 * @author hankcs
 */
public class PerceptronLexicalAnalyzer extends AbstractLexicalAnalyzer
{
    public PerceptronLexicalAnalyzer(PerceptronSegmenter segmenter)
    {
        super(segmenter);
    }

    public PerceptronLexicalAnalyzer(PerceptronSegmenter segmenter, PerceptronPOSTagger posTagger)
    {
        super(segmenter, posTagger);
    }

    public PerceptronLexicalAnalyzer(PerceptronSegmenter segmenter, PerceptronPOSTagger posTagger, PerceptronNERecognizer neRecognizer)
    {
        super(segmenter, posTagger, neRecognizer);
    }

    public PerceptronLexicalAnalyzer(LinearModel cwsModel, LinearModel posModel, LinearModel nerModel)
    {
        segmenter = new PerceptronSegmenter(cwsModel);
        if (posModel != null)
        {
            this.posTagger = new PerceptronPOSTagger(posModel);
            config.speechTagging = true;
        }
        else
        {
            this.posTagger = null;
        }
        if (nerModel != null)
        {
            neRecognizer = new PerceptronNERecognizer(nerModel);
            config.ner = true;
        }
        else
        {
            neRecognizer = null;
        }
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile, String posModelFile, String nerModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), posModelFile == null ? null : new LinearModel(posModelFile), nerModelFile == null ? null : new LinearModel(nerModelFile));
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile, String posModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), posModelFile == null ? null : new LinearModel(posModelFile), null);
    }

    public PerceptronLexicalAnalyzer(String cwsModelFile) throws IOException
    {
        this(new LinearModel(cwsModelFile), null, null);
    }

    public PerceptronLexicalAnalyzer(LinearModel CWSModel)
    {
        this(CWSModel, null, null);
    }

    /**
     * 加载配置文件指定的模型构造词法分析器
     *
     * @throws IOException
     */
    public PerceptronLexicalAnalyzer() throws IOException
    {
        this(HanLP.Config.PerceptronCWSModelPath, HanLP.Config.PerceptronPOSModelPath, HanLP.Config.PerceptronNERModelPath);
    }

    /**
     * 中文分词
     *
     * @param text
     * @param output
     */
    public void segment(String text, List<String> output)
    {
        String normalized = CharTable.convert(text);
        segment(text, normalized, output);
    }

    /**
     * 词性标注
     *
     * @param wordList
     * @return
     */
    public String[] partOfSpeechTag(List<String> wordList)
    {
        if (posTagger == null)
        {
            throw new IllegalStateException("未提供词性标注模型");
        }
        return tag(wordList);
    }

    /**
     * 命名实体识别
     *
     * @param wordArray
     * @param posArray
     * @return
     */
    public String[] namedEntityRecognize(String[] wordArray, String[] posArray)
    {
        if (neRecognizer == null)
        {
            throw new IllegalStateException("未提供命名实体识别模型");
        }
        return recognize(wordArray, posArray);
    }

    /**
     * 在线学习
     *
     * @param segmentedTaggedSentence 已分词、标好词性和命名实体的人民日报2014格式的句子
     * @return 是否学习成果（失败的原因是句子格式不合法）
     */
    public boolean learn(String segmentedTaggedSentence)
    {
        Sentence sentence = Sentence.create(segmentedTaggedSentence);
        return learn(sentence);
    }

    /**
     * 在线学习
     *
     * @param sentence 已分词、标好词性和命名实体的人民日报2014格式的句子
     * @return 是否学习成果（失败的原因是句子格式不合法）
     */
    public boolean learn(Sentence sentence)
    {
        CharTable.normalize(sentence);
        if (!getPerceptronSegmenter().learn(sentence)) return false;
        if (posTagger != null && !getPerceptronPOSTagger().learn(sentence)) return false;
        if (neRecognizer != null && !getPerceptionNERecognizer().learn(sentence)) return false;
        return true;
    }

    /**
     * 获取分词器
     *
     * @return
     */
    public PerceptronSegmenter getPerceptronSegmenter()
    {
        return (PerceptronSegmenter) segmenter;
    }

    /**
     * 获取词性标注器
     *
     * @return
     */
    public PerceptronPOSTagger getPerceptronPOSTagger()
    {
        return (PerceptronPOSTagger) posTagger;
    }

    /**
     * 获取命名实体识别器
     *
     * @return
     */
    public PerceptronNERecognizer getPerceptionNERecognizer()
    {
        return (PerceptronNERecognizer) neRecognizer;
    }

}