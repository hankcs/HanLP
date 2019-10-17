/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-25 3:44 PM</create-date>
 *
 * <copyright file="FeatureEngineering.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch05;

import com.hankcs.hanlp.corpus.MSR;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.model.perceptron.CWSTrainer;
import com.hankcs.hanlp.model.perceptron.PerceptronSegmenter;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.CWSInstance;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.utility.Utility;

import java.io.IOException;
import java.util.List;

/**
 * 《自然语言处理入门》5.6.7 中文分词特征工程
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class FeatureEngineering
{
    public static void main(String[] args) throws IOException
    {
        CWSTrainer trainer = new CWSTrainer()
        {
            @Override
            protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
            {
                return createMyCWSInstance(sentence, featureMap);
            }
        };
        LinearModel model = trainer.train(MSR.TRAIN_PATH, MSR.MODEL_PATH).getModel();
//        LinearModel model = new LinearModel(MSR.MODEL_PATH);
        PerceptronSegmenter segmenter = new PerceptronSegmenter(model)
        {
            @Override
            protected Instance createInstance(Sentence sentence, FeatureMap featureMap)
            {
                return createMyCWSInstance(sentence, featureMap);
            }
        };
        System.out.println(segmenter.segment("叠字特征帮助识别张文文李冰冰"));
    }

    private static Instance createMyCWSInstance(Sentence sentence, FeatureMap mutableFeatureMap)
    {
        List<Word> wordList = sentence.toSimpleWordList();
        String[] termArray = Utility.toWordArray(wordList);
        Instance instance = new MyCWSInstance(termArray, mutableFeatureMap);
        return instance;
    }

    /**
     * @author hankcs
     */
    public static class MyCWSInstance extends CWSInstance
    {
        @Override
        protected int[] extractFeature(String sentence, FeatureMap featureMap, int position)
        {
            int[] defaultFeatures = super.extractFeature(sentence, featureMap, position);
            char preChar = position >= 1 ? sentence.charAt(position - 1) : '_';
            String myFeature = preChar == sentence.charAt(position) ? "Y" : "N"; // 叠字特征
            int id = featureMap.idOf(myFeature);
            if (id != -1)
            {// 将叠字特征放到默认特征向量的尾部
                int[] newFeatures = new int[defaultFeatures.length + 1];
                System.arraycopy(defaultFeatures, 0, newFeatures, 0, defaultFeatures.length);
                newFeatures[defaultFeatures.length] = id;
                return newFeatures;
            }
            return defaultFeatures;
        }

        public MyCWSInstance(String[] termArray, FeatureMap featureMap)
        {
            super(termArray, featureMap);
        }

        public MyCWSInstance(String sentence, FeatureMap featureMap)
        {
            super(sentence, featureMap);
        }
    }
}
