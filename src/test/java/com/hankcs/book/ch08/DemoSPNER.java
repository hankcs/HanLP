/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-28 11:36 PM</create-date>
 *
 * <copyright file="DemoSPNER.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.hmm.HMMNERecognizer;
import com.hankcs.hanlp.model.perceptron.*;
import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;
import com.hankcs.hanlp.tokenizer.lexical.LexicalAnalyzer;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.IOException;

import static com.hankcs.book.ch08.DemoHMMNER.test;

/**
 * 《自然语言处理入门》8.5.3 基于感知机序列标注的命名实体识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoSPNER
{

    public static void main(String[] args) throws IOException
    {
        NERecognizer recognizer = train(PKU.PKU199801_TRAIN, PKU.NER_MODEL);
        test(recognizer);
        // 在线学习
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer(new PerceptronSegmenter(), new PerceptronPOSTagger(), (PerceptronNERecognizer) recognizer);//①
        Sentence sentence = Sentence.create("与/c 特朗普/nr 通/v 电话/n 讨论/v [太空/s 探索/vn 技术/n 公司/n]/nt");//②
        while (!analyzer.analyze(sentence.text()).equals(sentence))//③
            analyzer.learn(sentence);
    }

    public static NERecognizer train(String corpus, String model) throws IOException
    {
        if (IOUtil.isFileExisted(model))
            return new PerceptronNERecognizer(model);
        PerceptronTrainer trainer = new NERTrainer();
        return new PerceptronNERecognizer(trainer.train(corpus, corpus, model, 0, 50, 8).getModel());
    }
}
