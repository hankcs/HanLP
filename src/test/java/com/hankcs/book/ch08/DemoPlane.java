/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-29 8:49 PM</create-date>
 *
 * <copyright file="DemoPlane.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch08;

import com.hankcs.hanlp.model.perceptron.*;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.utility.TestUtility;

import java.io.IOException;

/**
 * 《自然语言处理入门》8.6.2 训练领域模型
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoPlane
{
    static String PLANE_CORPUS = TestUtility.ensureTestData("plane-re", "http://file.hankcs.com/corpus/plane-re.zip") + "/train.txt";
    static String PLANE_MODEL = PLANE_CORPUS.replace("train.txt", "model.bin");

    public static void main(String[] args) throws IOException
    {
        NERTrainer trainer = new NERTrainer();
        trainer.tagSet.nerLabels.clear(); // 不识别nr、ns、nt
        trainer.tagSet.nerLabels.add("np"); // 目标是识别np
        PerceptronNERecognizer recognizer = new PerceptronNERecognizer(trainer.train(PLANE_CORPUS, PLANE_MODEL).getModel());
        // 在NER预测前，需要一个分词器，最好训练自同源语料库
        LinearModel cwsModel = new CWSTrainer().train(PLANE_CORPUS, PLANE_MODEL.replace("model.bin", "cws.bin")).getModel();
        PerceptronSegmenter segmenter = new PerceptronSegmenter(cwsModel);
        PerceptronLexicalAnalyzer analyzer = new PerceptronLexicalAnalyzer(segmenter, new PerceptronPOSTagger(), recognizer);
        analyzer.enableTranslatedNameRecognize(false).enableCustomDictionary(false);
        System.out.println(analyzer.analyze("米高扬设计米格-17PF：米格-17PF型战斗机比米格-17P性能更好。"));
        System.out.println(analyzer.analyze("米格-阿帕奇-666S横空出世。"));    }
}
