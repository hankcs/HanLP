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

import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.crf.CRFNERecognizer;
import com.hankcs.hanlp.tokenizer.lexical.NERecognizer;

import java.io.IOException;

import static com.hankcs.book.ch08.DemoHMMNER.test;


/**
 * 《自然语言处理入门》8.5.4 基于条件随机场序列标注的命名实体识别
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoCRFNER
{
    public static void main(String[] args) throws IOException
    {
        NERecognizer recognizer = train(PKU.PKU199801_TRAIN, PKU.NER_MODEL);
        test(recognizer);
    }

    public static NERecognizer train(String corpus, String model) throws IOException
    {
        if (IOUtil.isFileExisted(model + ".txt")) // 若存在CRF++训练结果，则直接加载
            return new CRFNERecognizer(model + ".txt");
        CRFNERecognizer recognizer = new CRFNERecognizer(null); // 空白
        recognizer.train(corpus, model);
        return recognizer;
    }
}
