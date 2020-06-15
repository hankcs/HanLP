/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-29 4:18 PM</create-date>
 *
 * <copyright file="DemoCRFNER.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.crf.CRFNERecognizer;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.IOException;

import static com.hankcs.book.ch08.DemoPlane.PLANE_CORPUS;
import static com.hankcs.book.ch08.DemoPlane.PLANE_MODEL;


/**
 * 《自然语言处理入门》8.6.2 训练领域模型 （书本之外的补充试验）
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoCRFNERPlane
{
    public static void main(String[] args) throws IOException
    {
        NERecognizer recognizer = train(PLANE_CORPUS, PLANE_MODEL);
        String[] wordArray = {"歼", "-", "7", "战斗机", "正是", "仿照", "米格", "-", "21", "而", "制", "。"}; // 构造单词序列
        String[] posArray = {"v", "w", "w", "n", "d", "v", "nr", "w", "m", "c", "v", "w"}; // 构造词性序列
        String[] nerTagArray = recognizer.recognize(wordArray, posArray); // 序列标注
        for (int i = 0; i < wordArray.length; i++)
            System.out.printf("%-4s\t%s\t%s\t\n", wordArray[i], posArray[i], nerTagArray[i]);
    }

    public static NERecognizer train(String corpus, String model) throws IOException
    {
        if (IOUtil.isFileExisted(model + ".txt")) // 若存在CRF++训练结果，则直接加载
            return new CRFNERecognizer(model + ".txt");
        CRFNERecognizer recognizer = new CRFNERecognizer(null); // 空白
        recognizer.tagSet.nerLabels.clear(); // 不识别nr、ns、nt
        recognizer.tagSet.nerLabels.add("np"); // 目标是识别np
        recognizer.train(corpus, model);
        recognizer = new CRFNERecognizer(model + ".txt");
        return recognizer;
    }
}
