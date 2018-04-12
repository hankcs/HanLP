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

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.algorithm.MaxHeap;
import com.hankcs.hanlp.collection.trie.datrie.MutableDoubleArrayTrieInteger;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ByteArrayStream;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.feature.FeatureSortItem;
import com.hankcs.hanlp.model.perceptron.feature.ImmutableFeatureMDatMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;

import java.io.*;
import java.util.*;

import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

/**
 * 在线学习标注模型
 *
 * @author hankcs
 */
public class LinearModel implements ICacheAble
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

    /**
     * @param ratio 压缩比c（压缩掉的体积，压缩后体积变为1-c）
     * @return
     */
    public LinearModel compress(final double ratio)
    {
        if (ratio < 0 || ratio >= 1)
        {
            throw new IllegalArgumentException("压缩比必须介于 0 和 1 之间");
        }
        if (ratio == 0) return this;
        Set<Map.Entry<String, Integer>> featureIdSet = featureMap.entrySet();
        TagSet tagSet = featureMap.tagSet;
        MaxHeap<FeatureSortItem> heap = new MaxHeap<FeatureSortItem>((int) ((featureIdSet.size() - tagSet.sizeIncludingBos()) * (1.0f - ratio)), new Comparator<FeatureSortItem>()
        {
            @Override
            public int compare(FeatureSortItem o1, FeatureSortItem o2)
            {
                return Float.compare(o1.total, o2.total);
            }
        });

        for (Map.Entry<String, Integer> entry : featureIdSet)
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
        int size = items.size() + tagSet.sizeIncludingBos();
        float[] parameter = new float[size * tagSet.size()];
        MutableDoubleArrayTrieInteger mdat = new MutableDoubleArrayTrieInteger();
        for (Map.Entry<String, Integer> tag : tagSet)
        {
            mdat.add("BL=" + tag.getKey());
        }
        mdat.add("BL=_BL_");
        for (int i = 0; i < tagSet.size() * tagSet.sizeIncludingBos(); i++)
        {
            parameter[i] = this.parameter[i];
        }
        for (FeatureSortItem item : items)
        {
            int id = mdat.size();
            mdat.put(item.key, id);
            for (int i = 0; i < tagSet.size(); ++i)
            {
                parameter[id * tagSet.size() + i] = this.parameter[item.id * tagSet.size() + i];
            }
        }
        this.featureMap = new ImmutableFeatureMDatMap(mdat, tagSet);
        this.parameter = parameter;
        return this;
    }

    /**
     * 保存到路径
     *
     * @param modelFile
     * @throws IOException
     */
    public void save(String modelFile) throws IOException
    {
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(IOUtil.newOutputStream(modelFile)));
        save(out);
        out.close();
    }

    /**
     * 压缩并保存
     *
     * @param modelFile 路径
     * @param ratio     压缩比c（压缩掉的体积，压缩后体积变为1-c）
     * @throws IOException
     */
    public void save(String modelFile, final double ratio) throws IOException
    {
        save(modelFile, featureMap.entrySet(), ratio);
    }

    public void save(String modelFile, Set<Map.Entry<String, Integer>> featureIdSet, final double ratio) throws IOException
    {
        save(modelFile, featureIdSet, ratio, false);
    }

    /**
     * 保存
     *
     * @param modelFile    路径
     * @param featureIdSet 特征集（有些数据结构不支持遍历，可以提供构造时用到的特征集来规避这个缺陷）
     * @param ratio        压缩比
     * @param text         是否输出文本以供调试
     * @throws IOException
     */
    public void save(String modelFile, Set<Map.Entry<String, Integer>> featureIdSet, final double ratio, boolean text) throws IOException
    {
        float[] parameter = this.parameter;
        this.compress(ratio);

        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(IOUtil.newOutputStream(modelFile)));
        save(out);
        out.close();

        if (text)
        {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(IOUtil.newOutputStream(modelFile + ".txt"), "UTF-8"));
            TagSet tagSet = featureMap.tagSet;
            for (Map.Entry<String, Integer> entry : featureIdSet)
            {
                bw.write(entry.getKey());
                for (int i = 0; i < tagSet.size(); ++i)
                {
                    bw.write("\t");
                    bw.write(String.valueOf(parameter[entry.getValue() * tagSet.size() + i]));
                }
                bw.newLine();
            }
            bw.close();
        }
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
        if (HanLP.Config.DEBUG)
            logger.start("加载 %s ... ", modelFile);
        ByteArrayStream byteArray = ByteArrayStream.createByteArrayStream(modelFile);
        if (!load(byteArray))
        {
            throw new IOException(String.format("%s 加载失败", modelFile));
        }
        if (HanLP.Config.DEBUG)
            logger.finish(" 加载完毕\n");
    }

    public TagSet tagSet()
    {
        return featureMap.tagSet;
    }

    @Override
    public void save(DataOutputStream out) throws IOException
    {
        if (!(featureMap instanceof ImmutableFeatureMDatMap))
        {
            featureMap = new ImmutableFeatureMDatMap(featureMap.entrySet(), tagSet());
        }
        featureMap.save(out);
        for (float aParameter : this.parameter)
        {
            out.writeFloat(aParameter);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null)
            return false;
        featureMap = new ImmutableFeatureMDatMap();
        featureMap.load(byteArray);
        int size = featureMap.size();
        TagSet tagSet = featureMap.tagSet;
        parameter = new float[size * tagSet.size()];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < tagSet.size(); ++j)
            {
                parameter[i * tagSet.size() + j] = byteArray.nextFloat();
            }
        }
        assert !byteArray.hasMore();
        byteArray.close();
        return true;
    }
}
