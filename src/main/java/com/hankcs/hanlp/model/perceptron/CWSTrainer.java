/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM4:48</create-date>
 *
 * <copyright file="PerceptronSegmentTrainer.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.CWSInstance;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.model.perceptron.tagset.CWSTagSet;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * 感知机分词器训练工具
 *
 * @author hankcs
 */
public class CWSTrainer extends PerceptronTrainer
{
    @Override
    protected TagSet createTagSet()
    {
        return new CWSTagSet();
    }

    @Override
    protected Instance createInstance(Sentence sentence, FeatureMap mutableFeatureMap)
    {
        List<Word> wordList = Utility.toSimpleWordList(sentence);
        String[] termArray = Utility.toWordArray(wordList);
        Instance instance = new CWSInstance(termArray, mutableFeatureMap);
        return instance;
    }

    @Override
    public double[] evaluate(String developFile, LinearModel model) throws IOException
    {
        PerceptronLexicalAnalyzer segment = new PerceptronLexicalAnalyzer(model);
        double[] prf = Utility.prf(evaluate(developFile, segment));
        return prf;
    }

    private int[] evaluate(String developFile, final PerceptronLexicalAnalyzer segment) throws IOException
    {
        // int goldTotal = 0, predTotal = 0, correct = 0;
        final int[] stat = new int[3];
        Arrays.fill(stat, 0);
        loadInstance(developFile, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                List<Word> wordList = Utility.toSimpleWordList(sentence);
                String[] wordArray = Utility.toWordArray(wordList);
                stat[0] += wordArray.length;
                String text = com.hankcs.hanlp.utility.TextUtility.combine(wordArray);
                String[] predArray = segment.segment(text).toArray(new String[0]);
                stat[1] += predArray.length;

                int goldIndex = 0, predIndex = 0;
                int goldLen = 0, predLen = 0;

                while (goldIndex < wordArray.length && predIndex < predArray.length)
                {
                    if (goldLen == predLen)
                    {
                        if (wordArray[goldIndex].equals(predArray[predIndex]))
                        {
                            stat[2]++;
                            goldLen += wordArray[goldIndex].length();
                            predLen += wordArray[goldIndex].length();
                            goldIndex++;
                            predIndex++;
                        }
                        else
                        {
                            goldLen += wordArray[goldIndex].length();
                            predLen += predArray[predIndex].length();
                            goldIndex++;
                            predIndex++;
                        }
                    }
                    else if (goldLen < predLen)
                    {
                        goldLen += wordArray[goldIndex].length();
                        goldIndex++;
                    }
                    else
                    {
                        predLen += predArray[predIndex].length();
                        predIndex++;
                    }
                }

                return false;
            }
        });
        return stat;
    }
}
