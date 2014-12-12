/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 20:53</create-date>
 *
 * <copyright file="Model.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class CRFModel implements ICacheAble
{
    /**
     * 标签和id的相互转换
     */
    Map<String, Integer> tag2id;
    protected String[] id2tag;
    DoubleArrayTrie<FeatureFunction> featureFunctionTrie;
    List<FeatureTemplate> featureTemplateList;
    /**
     * tag的转移矩阵
     */
    protected double[][] matrix;

    public CRFModel()
    {
    }

    protected void onLoadTxtFinished()
    {
        // do no thing
    }

    public static CRFModel loadTxt(String path, CRFModel instance)
    {
        CRFModel CRFModel = instance;
        // 先尝试从bin加载
        if (CRFModel.load(ByteArray.createByteArray(path + Predefine.BIN_EXT))) return CRFModel;
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(path);
        if (!lineIterator.hasNext()) return null;
        logger.info(lineIterator.next());   // verson
        logger.info(lineIterator.next());   // cost-factor
        int maxid = Integer.parseInt(lineIterator.next().substring("maxid:".length()).trim());
        logger.info(lineIterator.next());   // xsize
        lineIterator.next();    // blank
        String line;
        int id = 0;
        CRFModel.tag2id = new HashMap<>();
        while ((line = lineIterator.next()).length() != 0)
        {
            CRFModel.tag2id.put(line, id);
            ++id;
        }
        CRFModel.id2tag = new String[CRFModel.tag2id.size()];
        final int size = CRFModel.id2tag.length;
        for (Map.Entry<String, Integer> entry : CRFModel.tag2id.entrySet())
        {
            CRFModel.id2tag[entry.getValue()] = entry.getKey();
        }
        TreeMap<String, FeatureFunction> featureFunctionMap = new TreeMap<>();  // 构建trie树的时候用
        List<FeatureFunction> featureFunctionList = new LinkedList<>(); // 读取权值的时候用
        CRFModel.featureTemplateList = new LinkedList<>();
        while ((line = lineIterator.next()).length() != 0)
        {
            if (!"B".equals(line))
            {
                FeatureTemplate featureTemplate = FeatureTemplate.create(line);
                CRFModel.featureTemplateList.add(featureTemplate);
            }
            else
            {
                CRFModel.matrix = new double[size][size];
            }
        }

        if (CRFModel.matrix != null)
        {
            lineIterator.next();    // 0 B
        }

        while ((line = lineIterator.next()).length() != 0)
        {
            String[] args = line.split(" ", 2);
            char[] charArray = args[1].toCharArray();
            FeatureFunction featureFunction = new FeatureFunction(charArray, size);
            featureFunctionMap.put(args[1], featureFunction);
            featureFunctionList.add(featureFunction);
        }

        if (CRFModel.matrix != null)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    CRFModel.matrix[i][j] = Double.parseDouble(lineIterator.next());
                }
            }
        }

        for (FeatureFunction featureFunction : featureFunctionList)
        {
            for (int i = 0; i < size; i++)
            {
                featureFunction.w[i] = Double.parseDouble(lineIterator.next());
            }
        }
        if (lineIterator.hasNext())
        {
            logger.warning("文本读取有残留，可能会出问题！" + path);
        }
        lineIterator.close();
        logger.info("开始构建双数组trie树");
        CRFModel.featureFunctionTrie = new DoubleArrayTrie<>();
        CRFModel.featureFunctionTrie.build(featureFunctionMap);
        // 缓存bin
        try
        {
            logger.info("开始缓存"+ path + Predefine.BIN_EXT);
            DataOutputStream out = new DataOutputStream(new FileOutputStream(path + Predefine.BIN_EXT));
            CRFModel.save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("在缓存" + path + Predefine.BIN_EXT + "时发生错误" + TextUtility.exceptionToString(e));
        }
        CRFModel.onLoadTxtFinished();
        return CRFModel;
    }

    /**
     * 维特比后向算法标注
     * @param table
     */
    public void tag(Table table)
    {
        int size = table.size();
        double bestScore = 0;
        int bestTag = 0;
        int tagSize = id2tag.length;
        LinkedList<double[]> scoreList = computeScoreList(table, 0);    // 0位置命中的特征函数
        for (int i = 0; i < tagSize; ++i)   // -1位置的标签遍历
        {
            for (int j = 0; j < tagSize; ++j)   // 0位置的标签遍历
            {
                double curScore = computeScore(scoreList, j);
                if (matrix != null)
                {
                    curScore += matrix[i][j];
                }
                if (curScore > bestScore)
                {
                    bestScore = curScore;
                    bestTag = j;
                }
            }
        }
        table.setLast(0, id2tag[bestTag]);
        int preTag = bestTag;
        // 0位置打分完毕，接下来打剩下的
        for (int i = 1; i < size; ++i)
        {
            scoreList = computeScoreList(table, i);    // i位置命中的特征函数
            bestScore = Double.MIN_VALUE;
            for (int j = 0; j < tagSize; ++j)   // i位置的标签遍历
            {
                double curScore =  computeScore(scoreList, j);
                if (matrix != null)
                {
                    curScore += matrix[preTag][j];
                }
                if (curScore > bestScore)
                {
                    bestScore = curScore;
                    bestTag = j;
                }
            }
            table.setLast(i, id2tag[bestTag]);
            preTag = bestTag;
        }
    }

    public LinkedList<double[]> computeScoreList(Table table, int current)
    {
        LinkedList<double[]> scoreList = new LinkedList<>();
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            char[] o = featureTemplate.generateParameter(table, current);
            FeatureFunction featureFunction = featureFunctionTrie.get(o);
            if (featureFunction == null) continue;
            scoreList.add(featureFunction.w);
        }

        return scoreList;
    }

    /**
     * 给一系列特征函数结合tag打分
     * @param scoreList
     * @param tag
     * @return
     */
    protected static double computeScore(LinkedList<double[]> scoreList, int tag)
    {
        double score = 0;
        for (double[] w : scoreList)
        {
            score += w[tag];
        }
        return score;
    }

    @Override
    public void save(DataOutputStream out) throws Exception
    {
        out.writeInt(id2tag.length);
        for (String tag : id2tag)
        {
            out.writeUTF(tag);
        }
        FeatureFunction[] valueArray = featureFunctionTrie.getValueArray(new FeatureFunction[0]);
        out.writeInt(valueArray.length);
        for (FeatureFunction featureFunction : valueArray)
        {
            featureFunction.save(out);
        }
        featureFunctionTrie.save(out);
        out.writeInt(featureTemplateList.size());
        for (FeatureTemplate featureTemplate : featureTemplateList)
        {
            featureTemplate.save(out);
        }
        if (matrix != null)
        {
            out.writeInt(matrix.length);
            for (double[] line : matrix)
            {
                for (double v : line)
                {
                    out.writeDouble(v);
                }
            }
        }
        else
        {
            out.writeInt(0);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null) return false;
        int size = byteArray.nextInt();
        id2tag = new String[size];
        tag2id = new HashMap<>(size);
        for (int i = 0; i < id2tag.length; i++)
        {
            id2tag[i] = byteArray.nextUTF();
            tag2id.put(id2tag[i], i);
        }
        FeatureFunction[] valueArray = new FeatureFunction[byteArray.nextInt()];
        for (int i = 0; i < valueArray.length; i++)
        {
            valueArray[i] = new FeatureFunction();
            valueArray[i].load(byteArray);
        }
        featureFunctionTrie = new DoubleArrayTrie<>();
        featureFunctionTrie.load(byteArray, valueArray);
        size = byteArray.nextInt();
        featureTemplateList = new ArrayList<>(size);
        for (int i = 0; i < size; ++i)
        {
            FeatureTemplate featureTemplate = new FeatureTemplate();
            featureTemplate.load(byteArray);
            featureTemplateList.add(featureTemplate);
        }
        size = byteArray.nextInt();
        if (size == 0) return true;
        matrix = new double[size][size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i][j] = byteArray.nextDouble();
            }
        }

        return true;
    }

    public static CRFModel loadTxt(String path)
    {
        return loadTxt(path, new CRFModel());
    }
}
