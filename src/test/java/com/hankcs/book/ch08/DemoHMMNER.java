/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-27 8:52 PM</create-date>
 *
 * <copyright file="HMMNER.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.model.hmm.HMMNERecognizer;
import com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer;
import com.hankcs.hanlp.tokenizer.lexical.LexicalAnalyzer;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.IOException;
import java.util.Map;

/**
 * 《自然语言处理入门》8.5.2 基于隐马尔可夫模型序列标注的命名实体识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoHMMNER
{
    public static void main(String[] args) throws IOException
    {
        NERecognizer recognizer = train(PKU.PKU199801_TRAIN);
        test(recognizer);
    }

    public static NERecognizer train(String corpus) throws IOException
    {
        HMMNERecognizer recognizer = new HMMNERecognizer();
        recognizer.train(corpus); // data/test/pku98/199801-train.txt
        return recognizer;
    }

    public static void test(NERecognizer recognizer) throws IOException
    {
        String[] wordArray = {"华北", "电力", "公司"}; // 构造单词序列
        String[] posArray = {"ns", "n", "n"}; // 构造词性序列
        String[] nerTagArray = recognizer.recognize(wordArray, posArray); // 序列标注
        for (int i = 0; i < wordArray.length; i++)
            System.out.printf("%s\t%s\t%s\t\n", wordArray[i], posArray[i], nerTagArray[i]);
        AbstractLexicalAnalyzer analyzer = new AbstractLexicalAnalyzer(new PerceptronSegmenter(), new PerceptronPOSTagger(), recognizer);
        analyzer.enableCustomDictionary(false);
        System.out.println(analyzer.analyze("华北电力公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"));
        Map<String, double[]> scores = Utility.evaluateNER(recognizer, PKU.PKU199801_TEST);
        Utility.printNERScore(scores);
    }
}
