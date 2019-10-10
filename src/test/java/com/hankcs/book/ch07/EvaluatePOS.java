/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-07-05 1:43 PM</create-date>
 *
 * <copyright file="EvaluatePOS.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch07;

import com.hankcs.hanlp.corpus.PKU;
import com.hankcs.hanlp.dependency.nnparser.util.PosTagUtil;
import com.hankcs.hanlp.model.crf.CRFPOSTagger;
import com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel;
import com.hankcs.hanlp.model.hmm.HMMPOSTagger;
import com.hankcs.hanlp.model.hmm.HiddenMarkovModel;
import com.hankcs.hanlp.model.hmm.SecondOrderHiddenMarkovModel;
import com.hankcs.hanlp.model.perceptron.POSTrainer;
import com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger;
import com.hankcs.hanlp.model.perceptron.PerceptronTrainer;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;

import java.io.File;
import java.io.IOException;

/**
 * 《自然语言处理入门》7.3 序列标注模型应用于词性标注
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class EvaluatePOS
{
    static HMMPOSTagger trainHMM(String corpus, HiddenMarkovModel model) throws IOException
    {
        HMMPOSTagger tagger = new HMMPOSTagger(model);
        tagger.train(corpus);
        return tagger;
    }

    static PerceptronPOSTagger trainPerceptronPOS(String corpus) throws IOException
    {
        PerceptronTrainer trainer = new POSTrainer();
        LinearModel model = trainer.train(corpus, File.createTempFile("hanlp", "pos.bin").getAbsolutePath()).getModel();
        return new PerceptronPOSTagger(model);
    }

    static CRFPOSTagger trainCRFPOS(String corpus) throws IOException
    {
        CRFPOSTagger tagger = new CRFPOSTagger(null);
        String modelPath = "data/test/pku98/pos.bin";
        tagger.train(corpus, modelPath);
        // 或者加载CRF++训练得到的pos.bin.txt
//        return new CRFPOSTagger(modelPath + ".txt");
        return new CRFPOSTagger(modelPath);
    }

    public static void main(String[] args) throws IOException
    {
        System.out.printf("一阶HMM\t%.2f%%\n", PosTagUtil.evaluate(trainHMM(PKU.PKU199801_TRAIN, new FirstOrderHiddenMarkovModel()), PKU.PKU199801_TEST));
        System.out.printf("二阶HMM\t%.2f%%\n", PosTagUtil.evaluate(trainHMM(PKU.PKU199801_TRAIN, new SecondOrderHiddenMarkovModel()), PKU.PKU199801_TEST));
        System.out.printf("感知机\t%.2f%%\n", PosTagUtil.evaluate(trainPerceptronPOS(PKU.PKU199801_TRAIN), PKU.PKU199801_TEST));
        System.out.printf("CRF\t%.2f%%\n", PosTagUtil.evaluate(trainCRFPOS(PKU.PKU199801_TRAIN), PKU.PKU199801_TEST));
    }
}
