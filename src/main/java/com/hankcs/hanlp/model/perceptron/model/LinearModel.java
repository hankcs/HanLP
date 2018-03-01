/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-09-04 PM10:29</create-date>
 *
 * <copyright file="LinearModel.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron.model;

import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.common.TaskType;
import com.hankcs.hanlp.model.perceptron.feature.FeatureSortItem;
import com.hankcs.hanlp.model.perceptron.feature.ImmutableFeatureHashMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.tagset.CWSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.POSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.algorithm.MaxHeap;

import java.io.*;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.System.out;

/**
 * 在线学习标注模型
 *
 * @author hankcs
 */
public class LinearModel
{
    /**
     * 特征全集
     */
    public FeatureMap featureMap;
    /**
     * 特征权重
     */
    public float[] parameter;


    public LinearModel(FeatureMap featureMap, float[] parameter)
    {
        this.featureMap = featureMap;
        this.parameter = parameter;
    }

    public LinearModel(FeatureMap featureMap)
    {
        this.featureMap = featureMap;
        parameter = new float[featureMap.size() * featureMap.tagSet.size()];
    }

    public LinearModel(String modelFile) throws IOException
    {
        load(modelFile);
    }

    public void save(String modelFile, Map<String, Integer> featureIdMap, final double ratio) throws IOException
    {
        save(modelFile, featureIdMap, ratio, false);
    }

    public void save(String modelFile, Map<String, Integer> featureIdMap, final double ratio, boolean text) throws IOException
    {
        out.printf("Saving model to %s at compress ratio %.2f...\n", modelFile, ratio);
        if (ratio < 0 || ratio >= 1)
        {
            throw new IllegalArgumentException("the compression ratio must be between 0 and 1");
        }

        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelFile)));
        BufferedWriter bw = null;
        if (text) bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelFile + ".txt"), "UTF-8"));

        // 保存标注集
        TagSet tagSet = featureMap.tagSet;
        tagSet.save(out);

        if (ratio > 0)
        {
            MaxHeap<FeatureSortItem> heap = new MaxHeap<FeatureSortItem>((int) ((featureIdMap.size() - tagSet.sizeIncludingBos()) * (1.0f - ratio)), new Comparator<FeatureSortItem>()
            {
                @Override
                public int compare(FeatureSortItem o1, FeatureSortItem o2)
                {
                    return Float.compare(o1.total, o2.total);
                }
            });

            for (Map.Entry<String, Integer> entry : featureIdMap.entrySet())
            {
                if (entry.getValue() < tagSet.sizeIncludingBos())
                {
                    continue;
                }
                FeatureSortItem item = new FeatureSortItem(entry, this.parameter, tagSet.size());
                if (item.total < 1e-3f) continue;
                heap.add(item);
            }

            List<FeatureSortItem> items = heap.toList();
            out.writeInt(items.size());
            for (FeatureSortItem item : items)
            {
                out.writeUTF(item.key);
                if (text)
                {
                    bw.write(item.key);
                    bw.newLine();
                }
                for (int i = 0; i < tagSet.size(); ++i)
                {
                    out.writeFloat(this.parameter[item.id * tagSet.size() + i]);
                    if (text)
                    {
                        bw.write(String.valueOf(this.parameter[item.id * tagSet.size() + i]));
                        bw.newLine();
                    }
                }
            }
        }
        else
        {
            out.writeInt(featureIdMap.size() - tagSet.sizeIncludingBos());
            for (Map.Entry<String, Integer> entry : featureIdMap.entrySet())
            {
                if (entry.getValue() < tagSet.sizeIncludingBos()) continue;
                out.writeUTF(entry.getKey());
                for (int i = 0; i < tagSet.size(); ++i)
                {
                    out.writeFloat(this.parameter[entry.getValue() * tagSet.size() + i]);
                }
            }

            if (text)
            {
                for (Map.Entry<String, Integer> entry : featureIdMap.entrySet())
                {
                    bw.write(entry.getKey());
                    bw.newLine();
                    for (int i = 0; i < tagSet.size(); ++i)
                    {
                        bw.write(String.valueOf(parameter[entry.getValue() * tagSet.size() + i]));
                        bw.newLine();
                    }
                }
            }
        }

        for (int i = 0; i < tagSet.size() * tagSet.sizeIncludingBos(); i++)
        {
            out.writeFloat(this.parameter[i]);
        }

        if (text) bw.close();
        out.close();
    }

    /**
     * 维特比解码
     *
     * @param instance 实例
     * @return
     */
    public double viterbiDecode(Instance instance)
    {
        return viterbiDecode(instance, instance.tagArray);
    }

    /**
     * 维特比解码
     *
     * @param instance   实例
     * @param guessLabel 输出标签
     * @return
     */
    public double viterbiDecode(Instance instance, int[] guessLabel)
    {
        final int[] allLabel = featureMap.allLabels();
        final int bos = featureMap.bosTag();
        final int sentenceLength = instance.tagArray.length;
        final int labelSize = allLabel.length;

        int[][] preMatrix = new int[sentenceLength][labelSize];
        double[][] scoreMatrix = new double[2][labelSize];

        for (int i = 0; i < sentenceLength; i++)
        {
            int _i = i & 1;
            int _i_1 = 1 - _i;
            int[] allFeature = instance.getFeatureAt(i);
            final int transitionFeatureIndex = allFeature.length - 1;
            if (0 == i)
            {
                allFeature[transitionFeatureIndex] = bos;
                for (int j = 0; j < allLabel.length; j++)
                {
                    preMatrix[0][j] = j;

                    double score = score(allFeature, j);

                    scoreMatrix[0][j] = score;
                }
            }
            else
            {
                for (int curLabel = 0; curLabel < allLabel.length; curLabel++)
                {

                    double maxScore = Integer.MIN_VALUE;

                    for (int preLabel = 0; preLabel < allLabel.length; preLabel++)
                    {

                        allFeature[transitionFeatureIndex] = preLabel;
                        double score = score(allFeature, curLabel);

                        double curScore = scoreMatrix[_i_1][preLabel] + score;

                        if (maxScore < curScore)
                        {
                            maxScore = curScore;
                            preMatrix[i][curLabel] = preLabel;
                            scoreMatrix[_i][curLabel] = maxScore;
                        }
                    }
                }

            }
        }

        int maxIndex = 0;
        double maxScore = scoreMatrix[(sentenceLength - 1) & 1][0];

        for (int index = 1; index < allLabel.length; index++)
        {
            if (maxScore < scoreMatrix[(sentenceLength - 1) & 1][index])
            {
                maxIndex = index;
                maxScore = scoreMatrix[(sentenceLength - 1) & 1][index];
            }
        }

        for (int i = sentenceLength - 1; i >= 0; --i)
        {
            guessLabel[i] = allLabel[maxIndex];
            maxIndex = preMatrix[i][maxIndex];
        }

        return maxScore;
    }

    /**
     * 通过命中的特征函数计算得分
     *
     * @param featureVector 压缩形式的特征id构成的特征向量
     * @return
     */
    public double score(int[] featureVector, int currentTag)
    {
        double score = 0;
        for (int index : featureVector)
        {
            if (index == -1)
            {
                continue;
            }
            else if (index < -1 || index >= featureMap.size())
            {
                throw new IllegalArgumentException("在打分时传入了非法的下标");
            }
            else
            {
                index = index * featureMap.tagSet.size() + currentTag;
                score += parameter[index];    // 其实就是特征权重的累加
            }
        }
        return score;
    }

    /**
     * 加载模型
     *
     * @param modelFile
     * @throws IOException
     */
    public void load(String modelFile) throws IOException
    {
        DataInputStream in = new DataInputStream(new FileInputStream(modelFile));
        TaskType type = TaskType.values()[in.readInt()];
        TagSet tagSet = null;
        switch (type)
        {
            case CWS:
                tagSet = new CWSTagSet();
                break;
            case POS:
                tagSet = new POSTagSet();
                break;
            case NER:
                tagSet = new NERTagSet();
                break;
        }
        tagSet.load(in);
        int size = in.readInt();
        parameter = new float[size * tagSet.size() + tagSet.size() * (tagSet.size() + 1)];
        int id = 0;
        Map<String, Integer> featureIdMap = new HashMap<String, Integer>();
        for (Map.Entry<String, Integer> tag : tagSet)
        {
            featureIdMap.put("BL=" + tag.getKey(), id++);
        }
        featureIdMap.put("BL=_BL_", id++);
        for (int i = 0; i < size; i++)
        {
            String key = in.readUTF();
            featureIdMap.put(key, id);
            for (int j = 0; j < tagSet.size(); ++j)
            {
                parameter[id * tagSet.size() + j] = in.readFloat();
            }
            ++id;
        }
        this.featureMap = new ImmutableFeatureHashMap(featureIdMap, tagSet);
        for (int i = 0; i < tagSet.size() * tagSet.sizeIncludingBos(); ++i)
        {
            parameter[i] = in.readFloat();
        }
        assert in.available() == 0;
        in.close();
    }

    public TagSet tagSet()
    {
        return featureMap.tagSet;
    }
}
