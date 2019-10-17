/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-06 1:36 PM</create-date>
 *
 * <copyright file="CustomCorpusPOS.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch07;

import com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;
import com.hankcs.hanlp.utility.TestUtility;

import java.io.IOException;

import static com.hankcs.book.ch07.EvaluatePOS.trainPerceptronPOS;

/**
 * 《自然语言处理入门》7.4 自定义词性
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class CustomCorpusPOS
{
    /**
     * 诛仙语料库
     * Zhang, Meishan and Zhang, Yue and Che, Wanxiang and Liu, Ting
     * Type-Supervised Domain Adaptation for Joint Segmentation and POS-Tagging
     */
    public static final String ZHUXIAN = TestUtility.ensureTestData("zhuxian", "http://file.hankcs.com/corpus/zhuxian.zip") + "/train.txt";

    public static void main(String[] args) throws IOException
    {
        PerceptronPOSTagger posTagger = trainPerceptronPOS(ZHUXIAN); // 训练
        AbstractLexicalAnalyzer analyzer = new AbstractLexicalAnalyzer(new PerceptronSegmenter(), posTagger); // 包装
        System.out.println(analyzer.analyze("陆雪琪的天琊神剑不做丝毫退避，直冲而上，瞬间，这两道奇光异宝撞到了一起。")); // 分词+标注
    }
}
