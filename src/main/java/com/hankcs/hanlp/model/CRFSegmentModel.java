/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 16:56</create-date>
 *
 * <copyright file="Index.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.ITrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.model.crf.CRFModel;
import com.hankcs.hanlp.model.crf.FeatureFunction;
import com.hankcs.hanlp.model.crf.Table;

import java.util.LinkedList;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 静态CRF分词模型
 *
 * @author hankcs
 */
public final class CRFSegmentModel extends CRFModel
{
    public static CRFModel crfModel;

    static
    {
        logger.info("CRF分词模型正在加载 " + HanLP.Config.CRFSegmentModelPath);
        long start = System.currentTimeMillis();
        crfModel = CRFModel.loadTxt(HanLP.Config.CRFSegmentModelPath, new CRFSegmentModel(new BinTrie<FeatureFunction>()));
        if (crfModel == null)
        {
            String error = "CRF分词模型加载 " + HanLP.Config.CRFSegmentModelPath + " 失败，耗时 " + (System.currentTimeMillis() - start) + " ms";
            logger.severe(error);
            throw new IllegalArgumentException(error);
        }
        else
            logger.info("CRF分词模型加载 " + HanLP.Config.CRFSegmentModelPath + " 成功，耗时 " + (System.currentTimeMillis() - start) + " ms");
    }

    private final static int idM = crfModel.getTagId("M");
    private final static int idE = crfModel.getTagId("E");
    private final static int idS = crfModel.getTagId("S");

    /**
     * 单例包装静态模型，不允许构造实例
     */
    private CRFSegmentModel()
    {
    }

    /**
     * 以指定的trie树结构储存内部特征函数
     * @param featureFunctionTrie
     */
    public CRFSegmentModel(ITrie<FeatureFunction> featureFunctionTrie)
    {
        super(featureFunctionTrie);
    }

    @Override
    public void tag(Table table)
    {
        int size = table.size();
        if (size == 1)
        {
            table.setLast(0, "S");
            return;
        }
        double[][] net = new double[size][4];
        for (int i = 0; i < size; ++i)
        {
            LinkedList<double[]> scoreList = computeScoreList(table, i);
            for (int tag = 0; tag < 4; ++tag)
            {
                net[i][tag] = computeScore(scoreList, tag);
            }
        }
        net[0][idM] = -1000.0;  // 第一个字不可能是M或E
        net[0][idE] = -1000.0;
        int[][] from = new int[size][4];
        for (int i = 1; i < size; ++i)
        {
            for (int now = 0; now < 4; ++now)
            {
                double maxScore = -1e10;
                for (int pre = 0; pre < 4; ++pre)
                {
                    double score = net[i - 1][pre] + matrix[pre][now] + net[i][now];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        from[i][now] = pre;
                    }
                }
                net[i][now] = maxScore;
            }
        }
        // 反向回溯最佳路径
        int maxTag = net[size - 1][idS] > net[size - 1][idE] ? idS : idE;
        table.setLast(size - 1, id2tag[maxTag]);
        maxTag = from[size - 1][maxTag];
        for (int i = size - 2; i > 0; --i)
        {
            table.setLast(i, id2tag[maxTag]);
            maxTag = from[i][maxTag];
        }
        table.setLast(0, id2tag[maxTag]);
    }
}
