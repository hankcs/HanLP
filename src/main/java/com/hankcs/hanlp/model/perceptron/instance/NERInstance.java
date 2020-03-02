/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-28 14:35</create-date>
 *
 * <copyright file="NERInstance.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.instance;

import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;
import com.hankcs.hanlp.model.perceptron.utility.Utility;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hankcs
 */
public class NERInstance extends Instance
{
    public NERInstance(String[] wordArray, String[] posArray, String[] nerArray, NERTagSet tagSet, FeatureMap featureMap)
    {
        this(wordArray, posArray, featureMap);

        tagArray = new int[wordArray.length];
        for (int i = 0; i < wordArray.length; i++)
        {
            tagArray[i] = tagSet.add(nerArray[i]);
        }
    }

    public NERInstance(String[][] tuples, NERTagSet tagSet, FeatureMap featureMap)
    {
        this(tuples[0], tuples[1], tuples[2], tagSet, featureMap);
    }

    public NERInstance(String[] wordArray, String[] posArray, FeatureMap featureMap)
    {
        initFeatureMatrix(wordArray, posArray, featureMap);
    }

    private void initFeatureMatrix(String[] wordArray, String[] posArray, FeatureMap featureMap)
    {
        featureMatrix = new int[wordArray.length][];
        for (int i = 0; i < featureMatrix.length; i++)
        {
            featureMatrix[i] = extractFeature(wordArray, posArray, featureMap, i);
        }
    }

    /**
     * 提取特征，override此方法来拓展自己的特征模板
     *
     * @param wordArray  词语
     * @param posArray   词性
     * @param featureMap 储存特征的结构
     * @param position   当前提取的词语所在的位置
     * @return 特征向量
     */
    protected int[] extractFeature(String[] wordArray, String[] posArray, FeatureMap featureMap, int position)
    {
        List<Integer> featVec = new ArrayList<Integer>();

        String pre2Word = position >= 2 ? wordArray[position - 2] : "_B_";
        String preWord = position >= 1 ? wordArray[position - 1] : "_B_";
        String curWord = wordArray[position];
        String nextWord = position <= wordArray.length - 2 ? wordArray[position + 1] : "_E_";
        String next2Word = position <= wordArray.length - 3 ? wordArray[position + 2] : "_E_";

        String pre2Pos = position >= 2 ? posArray[position - 2] : "_B_";
        String prePos = position >= 1 ? posArray[position - 1] : "_B_";
        String curPos = posArray[position];
        String nextPos = position <= posArray.length - 2 ? posArray[position + 1] : "_E_";
        String next2Pos = position <= posArray.length - 3 ? posArray[position + 2] : "_E_";

        StringBuilder sb = new StringBuilder();
        addFeatureThenClear(sb.append(pre2Word).append('1'), featVec, featureMap);
        addFeatureThenClear(sb.append(preWord).append('2'), featVec, featureMap);
        addFeatureThenClear(sb.append(curWord).append('3'), featVec, featureMap);
        addFeatureThenClear(sb.append(nextWord).append('4'), featVec, featureMap);
        addFeatureThenClear(sb.append(next2Word).append('5'), featVec, featureMap);
//        addFeatureThenClear(sb.append(pre2Word).append(preWord).append('6'), featVec, featureMap);
//        addFeatureThenClear(sb.append(preWord).append(curWord).append('7'), featVec, featureMap);
//        addFeatureThenClear(sb.append(curWord).append(nextWord).append('8'), featVec, featureMap);
//        addFeatureThenClear(sb.append(nextWord).append(next2Word).append('9'), featVec, featureMap);

        addFeatureThenClear(sb.append(pre2Pos).append('A'), featVec, featureMap);
        addFeatureThenClear(sb.append(prePos).append('B'), featVec, featureMap);
        addFeatureThenClear(sb.append(curPos).append('C'), featVec, featureMap);
        addFeatureThenClear(sb.append(nextPos).append('D'), featVec, featureMap);
        addFeatureThenClear(sb.append(next2Pos).append('E'), featVec, featureMap);
        addFeatureThenClear(sb.append(pre2Pos).append(prePos).append('F'), featVec, featureMap);
        addFeatureThenClear(sb.append(prePos).append(curPos).append('G'), featVec, featureMap);
        addFeatureThenClear(sb.append(curPos).append(nextPos).append('H'), featVec, featureMap);
        addFeatureThenClear(sb.append(nextPos).append(next2Pos).append('I'), featVec, featureMap);

        return toFeatureArray(featVec);
    }

    public NERInstance(String segmentedTaggedNERSentence, FeatureMap featureMap)
    {
        this(Sentence.create(segmentedTaggedNERSentence), featureMap);
    }

    public NERInstance(Sentence sentence, FeatureMap featureMap)
    {
        this(convertSentenceToArray(sentence, featureMap), (NERTagSet) featureMap.tagSet, featureMap);
    }

    private static String[][] convertSentenceToArray(Sentence sentence, FeatureMap featureMap)
    {
        NERTagSet tagSet = (NERTagSet) featureMap.tagSet;
        List<String[]> collector = Utility.convertSentenceToNER(sentence, tagSet);
        String[][] tuples = new String[3][collector.size()];
        String[] wordArray = tuples[0];
        String[] posArray = tuples[1];
        String[] tagArray = tuples[2];
        Utility.reshapeNER(collector, wordArray, posArray, tagArray);
        return tuples;
    }
}
